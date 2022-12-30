import numpy as np
import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    """
    q, k, v, dを受け取り，
    softmaxt(q k^T / \sqrt(d)) v を返す
    """
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        scalar = np.sqrt(self.d_k)
        attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar

        # maskがあれば，次元があっていないならそもそもエラー
        # maskは一般的？
        if mask is not None:
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )

            # torch.floatの最小値で指定範囲(mask)をマスクする
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max
        )
        # もし(batch_size, seq_size, column)なら，列単位でsoftmax?
        attention_weight = nn.functional.softmax(attention_weight, dim=2)
        return torch.matmul(attention_weight, v)
