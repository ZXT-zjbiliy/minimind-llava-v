import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from model.model_minimind import MiniMindForCausalLM
from model.projector_minimind_llava import MiniMindLlavaProjector


class MiniMindLlavaVLM(nn.Module):
    """
    Minimal multimodal wrapper:
    MiniMind LLM + LLaVA vision tower + a new projector.

    This file is the main integration point for your new project.

    Where to write the new projector:
        model/projector_minimind_llava.py

    What still needs to be plugged in:
    - a real LLaVA vision tower instance
    - pretrained vision checkpoint loading
    - multimodal dataset / collator
    """

    def __init__(
        self,
        llm: MiniMindForCausalLM,
        vision_tower: nn.Module,
        projector: nn.Module | None = None,
        image_token_id: int = 12,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.vision_tower = vision_tower
        self.projector = projector or MiniMindLlavaProjector(
            vision_hidden_size=1024,
            llm_hidden_size=llm.config.hidden_size,
        )
        self.image_token_id = image_token_id

    def encode_images(
        self,
        images: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # TorchDynamo is brittle around the vision branch because the number of
        # image tokens depends on runtime image-grid metadata.
        if torch.compiler.is_compiling():
            return self._encode_images_eager(images, image_grid_thw)
        vision_features, _ = self.vision_tower(images, image_grid_thw)
        image_embeddings = self.projector(vision_features)
        return image_embeddings

    @torch._dynamo.disable
    def _encode_images_eager(
        self,
        images: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        vision_features, _ = self.vision_tower(images, image_grid_thw)
        image_embeddings = self.projector(vision_features)
        return image_embeddings

    def merge_input_ids_with_image_features(
        self,
        input_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self._merge_input_ids_with_image_features_eager(
                input_ids=input_ids,
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
            )

        image_mask = (
            (input_ids == self.image_token_id)
            .unsqueeze(-1)
            .expand_as(text_embeddings)
            .to(text_embeddings.device)
        )

        n_image_tokens = (input_ids == self.image_token_id).sum().item()
        n_image_features = image_embeddings.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features {n_image_features} != image tokens {n_image_tokens}"
            )

        image_embeddings = image_embeddings.to(
            text_embeddings.device, text_embeddings.dtype
        )
        return text_embeddings.masked_scatter(image_mask, image_embeddings)

    @torch._dynamo.disable
    def _merge_input_ids_with_image_features_eager(
        self,
        input_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        image_mask = (
            (input_ids == self.image_token_id)
            .unsqueeze(-1)
            .expand_as(text_embeddings)
            .to(text_embeddings.device)
        )

        n_image_tokens = (input_ids == self.image_token_id).sum().item()
        n_image_features = image_embeddings.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features {n_image_features} != image tokens {n_image_tokens}"
            )

        image_embeddings = image_embeddings.to(
            text_embeddings.device, text_embeddings.dtype
        )
        return text_embeddings.masked_scatter(image_mask, image_embeddings)

    def forward_llm_backbone_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        This is a local multimodal copy of MiniMind's backbone forward path.
        It avoids changing the original copied model file on day 1.

        If you later want the cleaner long-term version, modify:
        - model/model_minimind.py -> MiniMindModel.forward
        - model/model_minimind.py -> MiniMindForCausalLM.forward
        - model/model_minimind.py -> MiniMindForCausalLM.generate
        so they all support inputs_embeds directly.
        """
        backbone = self.llm.model
        batch_size, seq_length, _ = inputs_embeds.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(backbone.layers)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = backbone.dropout(inputs_embeds)
        position_embeddings = (
            backbone.freqs_cos[start_pos : start_pos + seq_length],
            backbone.freqs_sin[start_pos : start_pos + seq_length],
        )

        presents = []
        for layer, past_key_value in zip(backbone.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = backbone.norm(hidden_states)
        aux_loss = sum(
            [
                layer.mlp.aux_loss
                for layer in backbone.layers
                if hasattr(layer.mlp, "aux_loss")
            ],
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
    ):
        text_embeddings = self.llm.model.embed_tokens(input_ids)

        if images is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required when images is not None")
            image_embeddings = self.encode_images(images, image_grid_thw)
            inputs_embeds = self.merge_input_ids_with_image_features(
                input_ids=input_ids,
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
            )
        else:
            inputs_embeds = text_embeddings

        hidden_states, past_key_values, aux_loss = self.forward_llm_backbone_with_embeds(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.llm.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
