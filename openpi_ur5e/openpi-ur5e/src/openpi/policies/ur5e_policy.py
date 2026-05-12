import numpy as np
import einops
import dataclasses
import logging

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import image_tools

logger = logging.getLogger("openpi")

_TARGET_IMAGE_HEIGHT = 224
_TARGET_IMAGE_WIDTH = 224

def make_ur5e_example() -> dict:
    """Creates a random input example for the UR5E policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(_TARGET_IMAGE_HEIGHT, _TARGET_IMAGE_WIDTH, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(_TARGET_IMAGE_HEIGHT, _TARGET_IMAGE_WIDTH, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(6),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class UR5EInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        joints = np.asarray(data["observation/joint_position"])
        joints = np.squeeze(joints)

        if joints.ndim != 1:
            raise ValueError(f"Expected joints to be 1D, got shape {joints.shape}")

        if "observation/gripper_position" in data:
            gripper_pos = np.asarray(data["observation/gripper_position"])
            if gripper_pos.ndim == 0:
                # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
                gripper_pos = gripper_pos[np.newaxis]
            state = np.concatenate([joints, gripper_pos])
        else:
            # Gripper position is embedded in the state or missing.
            if joints.shape[-1] == 7:
                state = joints
            elif joints.shape[-1] == 6:
                # Append zero placeholder for missing gripper value.
                state = np.concatenate([joints, np.zeros(1, dtype=joints.dtype)])
            else:
                raise ValueError(f"Unexpected joint/state shape: {joints.shape}")

        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image_left, wrist_image_right)
        
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                # We don't mask out padding images for FAST models.
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        logger.debug(f"Inputs: {inputs}")

        return inputs

@dataclasses.dataclass(frozen=True)
class UR5EDualCamInputs(transforms.DataTransformFn):
    """2-camera UR5E input transform (no duplicate wrist image).
    
    Maps only 2 physical cameras to model inputs:
      For PI0/PI05:  base_0_rgb (overhead) + left_wrist_0_rgb (wrist)
      For PI0_FAST:  base_0_rgb (overhead) + wrist_0_rgb (wrist)
    
    Saves ~25-30% compute vs 3-image version that duplicates the wrist camera.
    """
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        joints = np.asarray(data["observation/joint_position"])
        joints = np.squeeze(joints)

        if joints.ndim != 1:
            raise ValueError(f"Expected joints to be 1D, got shape {joints.shape}")

        if "observation/gripper_position" in data:
            gripper_pos = np.asarray(data["observation/gripper_position"])
            if gripper_pos.ndim == 0:
                gripper_pos = gripper_pos[np.newaxis]
            state = np.concatenate([joints, gripper_pos])
        else:
            if joints.shape[-1] == 7:
                state = joints
            elif joints.shape[-1] == 6:
                state = np.concatenate([joints, np.zeros(1, dtype=joints.dtype)])
            else:
                raise ValueError(f"Unexpected joint/state shape: {joints.shape}")

        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        # Only 2 images — model-specific key names
        match self.model_type:
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "wrist_0_rgb")
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb")
            case _:
                names = ("base_0_rgb", "left_wrist_0_rgb")

        images = (base_image, wrist_image)
        image_masks = (np.True_, np.True_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        logger.debug(f"Inputs (2-cam): {inputs}")
        return inputs


def make_ur5e_dual_cam_example() -> dict:
    """Creates a random input example for 2-camera UR5E policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(
            256, size=(_TARGET_IMAGE_HEIGHT, _TARGET_IMAGE_WIDTH, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(
            256, size=(_TARGET_IMAGE_HEIGHT, _TARGET_IMAGE_WIDTH, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(6),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class UR5EOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}