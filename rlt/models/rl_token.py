"""
RL Token Encoder-Decoder (PyTorch).

From the RLT paper (Physical Intelligence, 2026):
  "We train the VLA to produce an RL token that summarizes the VLA's internal
   representations. This RL token is then used as the input into a much smaller
   model that can be trained with RL in real time."

Architecture:
  Encoder: (N, 2048) VLM embeddings + <rl> query token → z_rl (token_dim,)
  Decoder: z_rl → reconstruct each of the N original tokens (teacher-forced)
  Loss:    MSE(decoded, stop_gradient(z_tokens))  — information bottleneck

The encoder compresses all VLM prefix tokens (image patches + language tokens)
into a single compact vector that retains enough information for the decoder
to reconstruct the originals. After training, only the encoder is used at
inference time — the decoder is discarded.

Usage:
    model = RLTokenModel(embed_dim=2048, token_dim=512)
    z = torch.randn(B, N, 2048)  # VLM embeddings from Pi05Hook
    loss, z_rl = model.compute_loss(z)  # training
    z_rl = model.extract(z)  # inference only (encoder path)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RLTokenModel(nn.Module):
    """RL Token encoder-decoder transformer.

    The encoder appends a learnable <rl> query token to the input sequence,
    processes everything through a transformer encoder, and extracts the
    output at the query position as the RL token.

    The decoder takes the RL token, expands it back, and autoregressively
    reconstructs each input token (teacher-forced during training).
    """

    def __init__(
        self,
        embed_dim: int = 2048,     # Gemma-2B hidden size — must match VLA
        token_dim: int = 512,      # compressed RL token output size
        enc_layers: int = 4,       # encoder transformer layers
        dec_layers: int = 4,       # decoder transformer layers
        n_heads: int = 8,          # attention heads
        ffn_dim: int = 2048,       # feedforward dimension
        max_len: int = 600,        # max N_prefix tokens (256*2_cams + lang ≈ 527)
        dropout: float = 0.0,      # no dropout (small dataset, short training)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_dim = token_dim

        # ── Encoder ────────────────────────────────────────────────────────
        # Learnable <rl> query token (appended at position N)
        self.rl_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # Positional embeddings for encoder (N input tokens + 1 query)
        self.enc_pos = nn.Embedding(max_len + 1, embed_dim)
        # Transformer encoder layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.enc_norm = nn.LayerNorm(embed_dim)
        # Project from embed_dim → token_dim (the bottleneck)
        self.to_token = nn.Linear(embed_dim, token_dim)

        # ── Decoder ────────────────────────────────────────────────────────
        # Expand token_dim back to embed_dim for decoder
        self.from_token = nn.Linear(token_dim, embed_dim)
        # Positional embeddings for decoder
        self.dec_pos = nn.Embedding(max_len, embed_dim)
        # Learnable BOS token for decoder (start of sequence)
        self.bos = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # Transformer decoder layers
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.dec_norm = nn.LayerNorm(embed_dim)
        # Output projection (reconstruct original embeddings)
        self.out_head = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode VLM embeddings into a single RL token.

        Args:
            z: (B, N, embed_dim) — VLM prefix token embeddings

        Returns:
            z_rl: (B, token_dim) — compressed RL token
        """
        B, N, _ = z.shape
        device = z.device

        # Add positional embeddings to input tokens
        pos_ids = torch.arange(N, device=device)
        x = z + self.enc_pos(pos_ids)  # (B, N, D)

        # Append <rl> query at position N
        rl_query = self.rl_query.expand(B, -1, -1)  # (B, 1, D)
        rl_pos = self.enc_pos(torch.tensor([N], device=device))  # (1, D)
        rl_query = rl_query + rl_pos
        x = torch.cat([x, rl_query], dim=1)  # (B, N+1, D)

        # Transformer encoder (full self-attention)
        x = self.encoder(x)  # (B, N+1, D)
        x = self.enc_norm(x)

        # Extract the output at the <rl> query position (last token)
        rl_output = x[:, -1, :]  # (B, D)

        # Project through bottleneck
        z_rl = self.to_token(rl_output)  # (B, token_dim)
        return z_rl

    def decode_tf(self, z_rl: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        """Teacher-forced decode: reconstruct original embeddings from z_rl.

        Args:
            z_rl: (B, token_dim) — compressed RL token
            z_tgt: (B, N, embed_dim) — target embeddings (for teacher forcing)

        Returns:
            z_hat: (B, N, embed_dim) — reconstructed embeddings
        """
        B, N, D = z_tgt.shape
        device = z_rl.device

        # Memory for cross-attention: expand z_rl to (B, 1, D)
        memory = self.from_token(z_rl).unsqueeze(1)  # (B, 1, D)

        # Teacher-forced target: shift right (BOS + z_tgt[:, :-1])
        bos = self.bos.expand(B, -1, -1)  # (B, 1, D)
        tgt = torch.cat([bos, z_tgt[:, :-1, :]], dim=1)  # (B, N, D)

        # Add positional embeddings
        pos_ids = torch.arange(N, device=device)
        tgt = tgt + self.dec_pos(pos_ids)

        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            N, device=device
        )

        # Transformer decoder with cross-attention to memory
        out = self.decoder(tgt, memory, tgt_mask=causal_mask)  # (B, N, D)
        out = self.dec_norm(out)

        # Project to output space
        z_hat = self.out_head(out)  # (B, N, D)
        return z_hat

    def compute_loss(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full training forward pass.

        Computes:
            1. Encode z → z_rl (gradient flows through encoder)
            2. Decode z_rl → z_hat (reconstruct against stop-gradient target)
            3. Loss = MSE(z_hat, stop_grad(z))

        Args:
            z: (B, N, embed_dim) — VLM prefix embeddings

        Returns:
            loss: scalar reconstruction loss
            z_rl: (B, token_dim) — compressed RL token (for logging)
        """
        z_sg = z.detach()  # Stop gradient on target (information bottleneck)
        z_rl = self.encode(z)  # Encode (gradient flows here)
        z_hat = self.decode_tf(z_rl, z_sg)  # Decode against sg target
        loss = F.mse_loss(z_hat, z_sg)
        return loss, z_rl

    @torch.no_grad()
    def extract(self, z: torch.Tensor) -> torch.Tensor:
        """Inference only: extract RL token without decoder.

        Args:
            z: (1, N, embed_dim) — single observation's VLM embeddings

        Returns:
            z_rl: (1, token_dim) — compressed RL token
        """
        return self.encode(z)

    def get_num_params(self) -> dict:
        """Count parameters by component."""
        enc_params = sum(
            p.numel() for n, p in self.named_parameters()
            if 'encoder' in n or 'enc_' in n or 'to_token' in n or 'rl_query' in n
        )
        dec_params = sum(
            p.numel() for n, p in self.named_parameters()
            if 'decoder' in n or 'dec_' in n or 'from_token' in n
            or 'out_head' in n or 'bos' in n
        )
        total = sum(p.numel() for p in self.parameters())
        return {
            "encoder": enc_params,
            "decoder": dec_params,
            "total": total,
        }
