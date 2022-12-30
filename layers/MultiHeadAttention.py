import torch
from layers.transformer.ScaledDotProductAttention import ScaledDotProductAttention
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h

        self.W_k = nn.Parameter(
            torch.Tensor(h, d_model, self.d_k)  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
        )

        self.W_q = nn.Parameter(
            torch.Tensor(h, d_model, self.d_k)  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
        )

        self.W_v = nn.Parameter(
            torch.Tensor(h, d_model, self.d_v)  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
        )

        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k)

        self.linear = nn.Linear(h * self.d_v, d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_3d: torch.Tensor = None,
    ) -> torch.Tensor:

        batch_size, seq_len = q.size(0), q.size(1)

        """repeat Query,Key,Value by num of heads"""
        # torch.repeatはコピーを複製, 0次元目からN次元目までを何倍するかを各引数に入れる
        # 今回はhead倍するだけ
        q = q.repeat(self.h, 1, 1, 1)  # head, batch_size, seq_len, d_model
        k = k.repeat(self.h, 1, 1, 1)  # head, batch_size, seq_len, d_model
        v = v.repeat(self.h, 1, 1, 1)  # head, batch_size, seq_len, d_model

        """Linear before scaled dot product attention"""
        # アインシュタイン縮約
        # head, batch_sizeに干渉しない行列演算を簡単に実装可能．．
        # https://qiita.com/TeamN/items/b9b1c065f866dab66db4 がわかりやすい
        q = torch.einsum(
            "hijk,hkl->hijl", (q, self.W_q)
        )  # head, batch_size, d_k, seq_len
        k = torch.einsum(
            "hijk,hkl->hijl", (k, self.W_k)
        )  # head, batch_size, d_k, seq_len
        v = torch.einsum(
            "hijk,hkl->hijl", (v, self.W_v)
        )  # head, batch_size, d_k, seq_len


        """Split heads"""
        # reshapeを行う関数だが，メモリ上に連続して並んでいないといけないという注意点が存在する点に注意
        q = q.view(self.h * batch_size, seq_len, self.d_k)
        k = k.view(self.h * batch_size, seq_len, self.d_k)
        v = v.view(self.h * batch_size, seq_len, self.d_v)

        if mask_3d is not None:
            mask_3d = mask_3d.repeat(self.h, 1, 1)

        """Scaled dot product attention"""
        attention_output = self.scaled_dot_product_attention(
            q, k, v, mask_3d
        )  # (head*batch_size, seq_len, d_model)

        # chunk（ぶつ切り）に分割する
        # self.h個で，分ける次元はdimで指定 
        attention_output = torch.chunk(attention_output, self.h, dim=0)
        attention_output = torch.cat(attention_output, dim=2)

        """Linear after scaled dot product attention"""
        output = self.linear(attention_output)
        return output
