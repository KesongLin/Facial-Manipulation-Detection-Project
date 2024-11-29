import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            num_layers: int,
            mlp_dim: int,
            dropout: float = 0.1
    ):
        super().__init__()

        # Create class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 17, embedding_dim))  # 16 + 1 for cls token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        # Create transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Get class token output and normalize
        x = self.norm(x)[:, 0]

        # Classification head
        x = self.head(x)
        return torch.sigmoid(x)