import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0) -> None:
        super().__init__()
        # nn.Embeddingで，語彙数，埋め込みの次元を指定することで単語埋め込みを実現してくれる
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_idx
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(input_batch)