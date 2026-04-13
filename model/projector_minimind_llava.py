import torch
from torch import nn


class MiniMindLlavaProjector(nn.Module):
    """
    Project LLaVA vision features into MiniMind hidden states.

    Expected input:
        vision_features: [num_patch_tokens, vision_hidden_size]

    Expected output:
        image_embeddings: [num_image_tokens, llm_hidden_size]

    Notes:
    - LLaVA-OneVision-1.5 4B uses vision_hidden_size=1024.
    - MiniMind default hidden_size is often 768.
    - LLaVA uses spatial_merge_size=2, so 4 patch tokens are merged into 1 image token.
    """

    def __init__(
        self,
        vision_hidden_size: int = 1024,
        llm_hidden_size: int = 768,
        spatial_merge_size: int = 2,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.merge_unit = spatial_merge_size ** 2
        self.merged_hidden_size = vision_hidden_size * self.merge_unit

        self.norm = nn.LayerNorm(vision_hidden_size)
        self.fc1 = nn.Linear(self.merged_hidden_size, self.merged_hidden_size)
        self.fc2 = nn.Linear(self.merged_hidden_size, llm_hidden_size)
        self.act = nn.GELU() if activation == "gelu" else nn.SiLU()

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        if vision_features.ndim != 2:
            raise ValueError(
                f"vision_features should be 2D [num_patch_tokens, hidden], got {tuple(vision_features.shape)}"
            )
        if vision_features.shape[-1] != self.vision_hidden_size:
            raise ValueError(
                f"Expected vision hidden size {self.vision_hidden_size}, got {vision_features.shape[-1]}"
            )
        if vision_features.shape[0] % self.merge_unit != 0:
            raise ValueError(
                "The number of vision tokens must be divisible by spatial_merge_size**2. "
                f"Got {vision_features.shape[0]} tokens and merge unit {self.merge_unit}."
            )

        x = self.norm(vision_features)
        x = x.reshape(-1, self.merged_hidden_size)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
