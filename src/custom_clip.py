from typing import Optional
from transformers import CLIPModel
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask


# modified from model/model.py

def encode_image(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return pooled_output, image_features


def encode_text_img_retrieval(self: CLIPModel, text, img_tokens, split_ind=4, repeat=True):
    # text.shape = [1, n_ctx]
    # img_tokens.shape = [batch_size, d_model]        
    if isinstance(img_tokens, tuple):
        b_size = img_tokens[0].shape[0]
    else:
        b_size = img_tokens.shape[0]
    if repeat:            
        text = text.repeat(b_size, 1)
    emblayer = self.text_model.embeddings
    x = emblayer.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model] # self.dtype=torch.float32
    collect_ind = text == 49407
    collect_ind = collect_ind.nonzero()[:, 1]
    ind_insert = text[0] == split_ind
    if isinstance(img_tokens, tuple):
        indexes = ind_insert.nonzero()
        for i, index in enumerate(indexes):
            img = img_tokens[i].view(b_size, 1, -1)
            x = torch.cat([x[:, :index], img, x[:, index+1:]], dim=1)
    else:
        img_tokens = img_tokens.view(b_size, 1, -1)
        ind_insert = ind_insert.nonzero()[0]
        x = torch.cat([x[:, :ind_insert], img_tokens, x[:, ind_insert+1:]], dim=1)
    #x = torch.cat([x, torch.zeros_like(x).cuda()[:, :1, :]], dim=1)
    # set output format
    output_attentions = self.config.output_attentions
    output_hidden_states = self.config.output_hidden_states
    return_dict = self.config.use_return_dict
    # set position embedding
    seq_length = x.shape[-1] if x is not None else x.shape[-2]
    position_ids = emblayer.position_ids[:, :seq_length]
    position_embeddings = emblayer.position_embedding(position_ids)
    hidden_states = x + position_embeddings
    # forward
    input_shape = x[:, :, 0].size()
    causal_attention_mask = _create_4d_causal_attention_mask(
        input_shape, hidden_states.dtype, device=hidden_states.device
    )
    # # expand attention_mask
    # if attention_mask is not None:
    #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #     attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
    encoder_outputs = self.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=None,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

    if self.text_model.eos_token_id == 2:
        # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
        # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
        # ------------------------------------------------------------
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            text.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
    else:
        # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            (text.to(dtype=torch.int, device=last_hidden_state.device) == self.text_model.eos_token_id)
            .int()
            .argmax(dim=-1),
        ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    text_outputs = BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )
    pooled_output = text_outputs[1]
    text_features = self.text_projection(pooled_output)
    # x = x + self.positional_embedding.type(self.dtype)
    # x = x.permute(1, 0, 2)  # NLD -> LND
    # x = self.transformer(x)
    # x = x.permute(1, 0, 2)  # LND -> NLD
    # x = self.ln_final(x).type(self.dtype)
    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)    
    return text_features