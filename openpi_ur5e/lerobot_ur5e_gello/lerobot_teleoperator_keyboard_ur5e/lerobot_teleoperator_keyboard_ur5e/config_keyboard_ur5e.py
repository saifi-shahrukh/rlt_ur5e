"""Configuration for the keyboard Cartesian teleoperator."""

from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard_ur5e")
@dataclass
class KeyboardUR5eConfig(TeleoperatorConfig):
    robot_ip: str = "172.22.1.139"
    mode: str = "cartesian"    # "cartesian" | "freedrive"
    trans_vel: float = 0.04    # m/s  — translation speed
    rot_vel: float = 0.3       # rad/s — rotation speed