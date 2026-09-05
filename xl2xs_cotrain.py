import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPTEmbedding, Block
from tokenizer import CharTokenizer


class Xl2XsGPTCoTrain(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        xs_n_embed,
        xl_n_embed,
        n_heads,
        n_layers,
        align_lambda,
        dropout=0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.align_lambda = align_lambda

        self.xs_transformer = nn.ModuleDict(
            {
                "embedding": GPTEmbedding(
                    vocab_size, self.block_size, xs_n_embed, dropout=dropout
                ),
                "blocks": nn.Sequential(
                    *[
                        Block(xs_n_embed, n_heads, self.block_size, dropout=dropout)
                        for _ in range(n_layers)
                    ]
                ),
                "ln_f": nn.LayerNorm(xs_n_embed),
            }
        )

        self.xs_lm_head = nn.Linear(xs_n_embed, vocab_size, bias=False)

        # small model weight tying
        self.xs_transformer["embedding"].token_embed.embedding.weight = (
            self.xs_lm_head.weight
        )

        self.xl_transformer = nn.ModuleDict(
            {
                "embedding": GPTEmbedding(
                    vocab_size, self.block_size, xl_n_embed, dropout=dropout
                ),
                "blocks": nn.Sequential(
                    *[
                        Block(xl_n_embed, n_heads, self.block_size, dropout=dropout)
                        for _ in range(n_layers)
                    ]
                ),
                "ln_f": nn.LayerNorm(xl_n_embed),
            }
        )

        self.xl_lm_head = nn.Linear(xl_n_embed, vocab_size, bias=False)

        # large model weight tying
        self.xl_transformer["embedding"].token_embed.embedding.weight = (
            self.xl_lm_head.weight
        )

        # Project xl_n_embed to xs_n_embed dimension for alignment loss caluclation.
        # self.align_head = nn.Linear(xl_n_embed, xs_n_embed, bias=False) # shape: [xs, xl]

        self.align_head = nn.Sequential(
            nn.Linear(xl_n_embed, xs_n_embed, bias=False), nn.LayerNorm(xs_n_embed)
        )

        # Initialize weights properly, otherwise the training will crash
        self.apply(self._init_weights)

        print(f"Model initialized - {self.count_params() / 1e6:.2f}M parameters")
        print(
            f"XS model parameters - {sum([p.numel() for p in self.xs_transformer.parameters()])/1e6:.2f}M"
        )
        print(
            f"XL model parameters - {sum([p.numel() for p in self.xl_transformer.parameters()])/1e6:.2f}M"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert (
            T <= self.block_size
        ), f"Sequence length {T} exceeds block_size {self.block_size}"

        xs = self.xs_transformer["embedding"](idx)
        xs = self.xs_transformer["blocks"](xs)
        xs = self.xs_transformer["ln_f"](xs)
        xs_logits = self.xs_lm_head(xs)

        xl = self.xl_transformer["embedding"](idx)
        xl = self.xl_transformer["blocks"](xl)
        xl = self.xl_transformer["ln_f"](xl)
        xl_logits = self.xl_lm_head(xl)

        assert (
            xs_logits.shape == xl_logits.shape
        ), f"XS logits shape ({xs.logits.shape}) != XL logits shape ({xl_logits.shape})"
        B, T, V = xs_logits.shape

        # start with aligning final residual streams
        xl2xs = self.align_head(xl)

        xs_loss = None
        xl_loss = None
        if targets is not None:
            xs_loss = F.cross_entropy(xs_logits.view(B * T, V), targets.view(B * T))
            xl_loss = F.cross_entropy(xl_logits.view(B * T, V), targets.view(B * T))

            # Don't propagate gradients to XS, as we want to anchor XL w.r.t. XS. Not
            # detaching XS will update XS model gradients too which may lead to
            # unknown behavior.
            align_loss = F.mse_loss(xs.detach(), xl2xs)

            total_loss = xs_loss + xl_loss + self.align_lambda * align_loss

            metrics = {
                "xs_loss": xs_loss,
                "xl_loss": xl_loss,
                "align_loss": align_loss,
                "total_loss": total_loss,
            }
            return metrics
