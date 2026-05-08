"""
π0.5 / π0 / π0-FAST PyTorch forward hook for VLM embedding extraction.

This module attaches a forward hook to the final layer norm of the PaliGemma
VLM backbone to capture the internal token representations (before the LM head).
These representations are then compressed by the RL Token encoder.

Supports all three model variants:
  - π0 (pi0): paligemma_with_expert.paligemma.language_model.model.norm
  - π0.5 (pi05): same location (paligemma backbone unchanged)
  - π0-FAST: same backbone, different action head (FAST tokenizer)

The hook captures the prefix tokens (image patches + language tokens) which
contain the VLM's understanding of the scene. Action tokens are discarded.

Architecture path:
  PI0Pytorch
  └── paligemma_with_expert (PaliGemmaWithExpertModel)
      ├── paligemma (PaliGemma)
      │   └── language_model (GemmaModel)  
      │       └── model
      │           └── norm  ← HOOK HERE (final RMSNorm, output: B×N_total×2048)
      └── gemma_expert (Gemma 300M action expert)

Usage:
    hook = Pi05Hook(
        checkpoint_dir="path/to/checkpoint",
        config_name="pi0_fast_ur5e_peg_insertion_lora",
    )
    z_tokens, action_chunk = hook.get_embeddings_and_actions(obs)
    # z_tokens: (N_prefix, 2048) — input to RLTokenModel
    # action_chunk: (H, 7) — VLA reference actions ã
    hook.close()
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


class Pi05Hook:
    """Wraps an OpenPI PyTorch π0/π0.5/π0-FAST model with a forward hook.

    Exposes:
        get_embeddings_and_actions(obs) → (z_tokens, action_chunk)
        get_action_only(obs)            → action_chunk
    """

    def __init__(
        self,
        checkpoint_dir: str,
        config_name: str = "pi0_fast_ur5e_peg_insertion_lora",
        device: str = "cuda",
        img_size: tuple[int, int] = (224, 224),
        openpi_src_path: Optional[str] = None,
    ):
        """
        Args:
            checkpoint_dir: Path to trained openpi checkpoint directory
            config_name: OpenPI training config name (from config.py)
            device: "cuda" or "cpu"
            img_size: Image size expected by SigLIP (224×224)
            openpi_src_path: Path to openpi source (auto-detected if None)
        """
        self.device = device
        self.img_size = img_size
        self._captured: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._model = None
        self._policy = None

        # Add openpi to path if needed
        if openpi_src_path:
            sys.path.insert(0, openpi_src_path)
        else:
            # Auto-detect from common locations
            candidates = [
                Path(__file__).parents[2] / "openpi_ur5e" / "openpi-ur5e" / "src",
                Path.home() / "ur5e_hande_workspace" / "rlt_ur5e" / "openpi_ur5e" / "openpi-ur5e" / "src",
            ]
            for c in candidates:
                if c.exists():
                    sys.path.insert(0, str(c))
                    break

        # Import openpi modules
        try:
            from openpi.training import config as _config
            from openpi.policies import policy_config
        except ImportError as e:
            raise ImportError(
                f"Cannot import openpi. Ensure openpi-ur5e/src is in PYTHONPATH. Error: {e}"
            ) from e

        # Load the trained policy
        print(f"[Pi05Hook] Loading config '{config_name}'...")
        cfg = _config.get_config(config_name)

        print(f"[Pi05Hook] Loading checkpoint from '{checkpoint_dir}'...")
        self._policy = policy_config.create_trained_policy(cfg, checkpoint_dir)

        # Access the underlying PyTorch model
        # The policy wraps the model differently depending on version
        if hasattr(self._policy, '_model'):
            self._model = self._policy._model
        elif hasattr(self._policy, 'model'):
            self._model = self._policy.model
        else:
            raise AttributeError(
                "Cannot find model attribute on policy. "
                "Check openpi policy implementation."
            )

        self._model.eval()
        if device != "cpu":
            self._model = self._model.to(device)

        # Determine action horizon from model config
        if hasattr(self._model, 'config'):
            self._action_horizon = getattr(self._model.config, 'action_horizon', 50)
        else:
            self._action_horizon = 50  # default

        # ── Attach hook to final PaliGemma layer norm ────────────────────
        target_module = self._find_norm_layer()
        self._hook_handle = target_module.register_forward_hook(self._hook_fn)
        print(f"[Pi05Hook] Hook attached. Action horizon: {self._action_horizon}")
        print(f"[Pi05Hook] Ready on device: {device}")

    def _find_norm_layer(self) -> torch.nn.Module:
        """Find the final norm layer in the PaliGemma backbone.

        Tries multiple paths to support different model versions.
        """
        # Path candidates (in order of preference)
        paths = [
            # π0.5 / π0 standard path
            "paligemma_with_expert.paligemma.language_model.model.norm",
            # Alternative: direct language_model path
            "paligemma_with_expert.paligemma.model.norm",
            # π0-FAST might have slightly different structure
            "paligemma_with_expert.paligemma.language_model.norm",
        ]

        for path in paths:
            try:
                module = self._model
                for attr in path.split("."):
                    module = getattr(module, attr)
                print(f"[Pi05Hook] Found norm layer at: {path}")
                return module
            except AttributeError:
                continue

        # Fallback: search for any module named 'norm' at the right depth
        print("[Pi05Hook] WARNING: Standard paths failed. Searching...")
        for name, module in self._model.named_modules():
            if name.endswith(".norm") and "language_model" in name:
                print(f"[Pi05Hook] Found norm layer at: {name}")
                return module

        raise RuntimeError(
            "Cannot find final norm layer in model. "
            "Available modules: " +
            str([n for n, _ in self._model.named_modules() if 'norm' in n][:10])
        )

    def _hook_fn(self, module, inp, out):
        """Forward hook callback — captures the output tensor."""
        # Handle tuple outputs (some norm layers return (output, gate))
        if isinstance(out, tuple):
            out = out[0]
        self._captured = out.detach().cpu()

    def _prep_obs(self, obs: dict) -> dict:
        """Convert observation dict to OpenPI inference format.

        Args:
            obs: dict with keys like:
                - "wrist_image" or "wrist_1": (H, W, 3) uint8
                - "base_image" or "overview": (H, W, 3) uint8
                - "proprio" or "state": (D,) float32

        Returns:
            OpenPI-formatted observation dict
        """
        # Handle different key naming conventions
        wrist_keys = ["wrist_image", "wrist_1", "wrist_cam"]
        base_keys = ["base_image", "overview", "overview_cam"]
        state_keys = ["proprio", "state", "tcp_pose"]

        wrist = None
        for k in wrist_keys:
            if k in obs:
                wrist = obs[k]
                break

        base = None
        for k in base_keys:
            if k in obs:
                base = obs[k]
                break

        state = None
        for k in state_keys:
            if k in obs:
                state = obs[k]
                break

        if wrist is None:
            raise KeyError(f"No wrist image found. Available keys: {list(obs.keys())}")

        # Handle stacked observations (1, H, W, C) → (H, W, C)
        if wrist.ndim == 4:
            wrist = wrist[0]
        if base is not None and base.ndim == 4:
            base = base[0]
        if state is not None and state.ndim == 2:
            state = state[0]

        # Resize images to SigLIP input size
        wrist_resized = cv2.resize(
            wrist.astype(np.uint8), (self.img_size[1], self.img_size[0])
        )

        result = {
            "observation/image": wrist_resized,
            "observation/wrist_image": wrist_resized,
        }

        if base is not None:
            base_resized = cv2.resize(
                base.astype(np.uint8), (self.img_size[1], self.img_size[0])
            )
            result["observation/image_1"] = base_resized

        if state is not None:
            result["observation/state"] = state.astype(np.float32)

        # Add language prompt
        result["prompt"] = "Pick up the peg and insert it into the hole."

        return result

    @torch.no_grad()
    def get_embeddings_and_actions(
        self, obs: dict, prompt: Optional[str] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get both VLM embeddings and action chunk from a single forward pass.

        Args:
            obs: observation dict (images + proprio)
            prompt: optional language instruction override

        Returns:
            z_tokens:     (N_prefix, 2048) — VLM last-layer token embeddings
            action_chunk: (H, 7)           — VLA reference action chunk ã
        """
        self._captured = None

        prepped = self._prep_obs(obs)
        if prompt:
            prepped["prompt"] = prompt

        # Run inference (triggers the forward hook)
        result = self._policy.infer(prepped)

        # Extract action chunk
        if isinstance(result, dict):
            action_chunk = np.array(result.get("actions", result.get("action", [])),
                                   dtype=np.float32)
        else:
            action_chunk = np.array(result, dtype=np.float32)

        # Reshape if needed (should be (H, action_dim))
        if action_chunk.ndim == 1:
            action_dim = 7  # UR5e: 6 joints + gripper
            action_chunk = action_chunk.reshape(-1, action_dim)

        # Extract captured embeddings
        if self._captured is None:
            raise RuntimeError(
                "Forward hook did not fire! Check model architecture. "
                "The hook should be on the final norm layer of PaliGemma."
            )

        z_all = self._captured[0].float().numpy()  # (N_total, 2048)

        # Remove action tokens (last H positions in the sequence)
        # For π0-FAST, the suffix tokens might be different
        # We take everything except the last action_horizon tokens
        n_total = z_all.shape[0]
        n_prefix = max(n_total - self._action_horizon, n_total // 2)
        z_tokens = z_all[:n_prefix]  # (N_prefix, 2048)

        return z_tokens, action_chunk

    @torch.no_grad()
    def get_action_only(self, obs: dict, prompt: Optional[str] = None) -> np.ndarray:
        """Fast path — get VLA action chunk without embedding extraction.

        Args:
            obs: observation dict
            prompt: optional language instruction

        Returns:
            action_chunk: (H, 7) VLA reference actions
        """
        prepped = self._prep_obs(obs)
        if prompt:
            prepped["prompt"] = prompt

        result = self._policy.infer(prepped)

        if isinstance(result, dict):
            action_chunk = np.array(result.get("actions", result.get("action", [])),
                                   dtype=np.float32)
        else:
            action_chunk = np.array(result, dtype=np.float32)

        if action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(-1, 7)

        return action_chunk

    def close(self):
        """Remove hook and free resources."""
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None
        self._captured = None
        print("[Pi05Hook] Hook removed, resources freed.")

    def __del__(self):
        self.close()
