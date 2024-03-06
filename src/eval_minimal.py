from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from tqdm import tqdm


from eval_utils import get_metrics_fashion
from params import parse_args
from src.data import FashionIQ
from src.eval_utils import evaluate_fashion
from blip_diff_pipeline import BlipDiffusionPipeline


def preprocess(n_px: int, is_train: bool):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            lambda image: image.convert('RGB'),
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert('RGB'),
            ToTensor(),
            normalize,
        ])


def load_model(args):
    model = BlipDiffusionPipeline.from_pretrained(
        "Salesforce/blipdiffusion", torch_dtype=torch.float16
    ).to("cuda")
    return preprocess(224, False), model


def fashion_eval(args, root_project):
    preprocess_val, model = load_model(args)
    assert args.source_data in ['dress', 'shirt', 'toptee']
    source_dataset = FashionIQ(cloth=args.source_data,
                               transforms=preprocess_val,
                               root=root_project,
                               is_return_target_path=True)
    target_dataset = FashionIQ(cloth=args.source_data,
                               transforms=preprocess_val,
                               root=root_project,
                               mode='imgs')
    source_dataloader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    target_dataloader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    evaluate_fashion(model, args, source_dataloader, target_dataloader)


def evaluate_fashion(model, args, source_loader, target_loader):
    all_composed_feat = []
    all_image_features = []
    all_target_paths = []
    all_answer_paths = []
    # get all image features
    with torch.no_grad():
        for batch in tqdm(target_loader, desc="Target Features:"):
            target_images, target_paths = batch
            target_images = target_images.cuda(0, non_blocking=True)
            image_features = model.encode_image(target_images)  # TODO inference alignment
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)
    # get composed features
    with torch.no_grad():
        for batch in tqdm(source_loader):
            ref_images, target_images, target_caption, caption_only, answer_paths, ref_names, captions = batch
            for path in answer_paths:
                all_answer_paths.append(path)
            composed_feat = model.forward()  # Tensor: [bs, 768] TODO inference alignment
            composed_feat = composed_feat / composed_feat.norm(dim=-1, keepdim=True)
            all_composed_feat.append(composed_feat)
    metric_func = partial(get_metrics_fashion,
                          image_features=torch.cat(all_image_features),
                          target_names=all_target_paths, answer_names=all_answer_paths)
    feats = {
        'composed': torch.cat(all_composed_feat)
    }
    for key, value in feats.items():
        metrics = metric_func(ref_features=value)
        print(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )
    return metrics


if __name__ == '__main__':
    args = parse_args()
    fashion_eval(args, "data")
