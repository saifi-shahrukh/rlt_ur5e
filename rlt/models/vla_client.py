"""VLA WebSocket Client — connects to existing OpenPI server for reference actions.

Instead of loading the VLA model in-process (which requires JAX + Python 3.11),
this client connects to the already-running OpenPI WebSocket server.

The server is started in a separate terminal:
    cd openpi_ur5e/openpi-ur5e
    .venv/bin/python scripts/serve_policy.py --port 8000 ...

This client sends observations and receives action chunks back.
No embedding extraction is possible via this interface (we use zero z_rl
or a pre-computed RL Token fallback).

Usage:
    client = VLAClient(server_url="ws://localhost:8000")
    actions = client.get_actions(obs_dict)
    client.close()
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    HAS_OPENPI_CLIENT = True
except ImportError:
    WebsocketClientPolicy = None
    HAS_OPENPI_CLIENT = False


class VLAClient:
    """Connects to OpenPI WebSocket server for VLA action inference.

    This replaces Pi05Hook for online RL — no need for JAX/OpenPI deps
    in the training process.
    """

    def __init__(
        self,
        server_url: str = "ws://localhost:8000",
        prompt: str = "Pick up the peg and insert it into the hole.",
        action_horizon: int = 30,
        action_dim: int = 7,
    ):
        self.server_url = server_url
        self.prompt = prompt
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self._client = None
        self._connected = False

        self._connect()

    def _connect(self):
        """Connect to the VLA server."""
        if not HAS_OPENPI_CLIENT:
            print("[VLAClient] openpi_client not installed.")
            print("[VLAClient] Install with: pip install openpi-client")
            self._connected = False
            return

        try:
            host = self.server_url.replace("ws://", "").split(":")[0]
            port = int(self.server_url.split(":")[-1])
            self._client = WebsocketClientPolicy(
                host=host,
                port=port,
            )
            self._connected = True
            print(f"[VLAClient] Connected to VLA server at {self.server_url}")
        except Exception as e:
            print(f"[VLAClient] Failed to connect: {e}")
            self._connected = False

    def get_actions(
        self,
        joint_position: np.ndarray,
        gripper_position: float,
        images: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Get VLA reference actions from the server.

        Args:
            joint_position: (6,) joint angles in radians
            gripper_position: scalar gripper value (0-1)
            images: dict of camera_name -> (H, W, 3) uint8 images
                Expected keys: 'exterior_image_1_left', 'wrist_image_left', 'wrist_image_right'

        Returns:
            actions: (action_horizon, action_dim) absolute joint targets
        """
        if not self._connected:
            # Return current state repeated (no movement)
            action = np.zeros((self.action_horizon, self.action_dim))
            action[:, :6] = joint_position
            action[:, 6] = gripper_position
            return action

        try:
            # Build observation dict matching what the server expects
            obs = {
                "observation/joint_position": np.array(joint_position, dtype=np.float32),
                "observation/gripper_position": np.array([gripper_position], dtype=np.float32),
                "prompt": self.prompt,
            }

            # Add images with OpenPI naming convention
            for key, img in images.items():
                obs[f"observation/{key}"] = img.astype(np.uint8)

            # Call server
            result = self._client.infer(obs)

            # Extract actions
            if isinstance(result, dict) and "actions" in result:
                actions = np.array(result["actions"], dtype=np.float32)
            else:
                actions = np.array(result, dtype=np.float32)

            # Ensure correct shape
            if actions.ndim == 1:
                actions = actions.reshape(-1, self.action_dim)

            return actions[:self.action_horizon]

        except Exception as e:
            print(f"[VLAClient] Inference failed: {e}")
            action = np.zeros((self.action_horizon, self.action_dim))
            action[:, :6] = joint_position
            action[:, 6] = gripper_position
            return action

    def is_connected(self) -> bool:
        return self._connected

    def close(self):
        """Close websocket connection."""
        if self._client is not None:
            try:
                self._client.close()
            except:
                pass
        self._connected = False
        print("[VLAClient] Connection closed.")
