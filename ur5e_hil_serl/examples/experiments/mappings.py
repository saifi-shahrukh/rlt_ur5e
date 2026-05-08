"""Task name → config mapping.

Includes both:
1. Our UR5e + Hand-E experiments (primary)
2. Original hil-serl Franka experiments (reference, will fail without Franka hardware)

The train_rlpd.py script uses this to find the right config from --exp_name.
"""

# ===== UR5e + Hand-E experiments (our hardware) =====
from experiments.peg_insertion.config import TrainConfig as PegInsertionTrainConfig
from experiments.pcb_insertion.config import TrainConfig as PCBInsertionTrainConfig
from experiments.bin_relocation.config import TrainConfig as BinRelocationTrainConfig
from experiments.cable_routing.config import TrainConfig as CableRoutingTrainConfig

# ===== Original hil-serl Franka experiments (reference) =====
# These require franka_env and Franka hardware — included for compatibility
try:
    from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
    from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
    from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
    from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
    _FRANKA_AVAILABLE = True
except ImportError:
    _FRANKA_AVAILABLE = False


CONFIG_MAPPING = {
    # UR5e tasks (our setup)
    "peg_insertion": PegInsertionTrainConfig,
    "pcb_insertion": PCBInsertionTrainConfig,
    "bin_relocation": BinRelocationTrainConfig,
    "cable_routing": CableRoutingTrainConfig,
}

# Add Franka experiments if available
if _FRANKA_AVAILABLE:
    CONFIG_MAPPING.update({
        "ram_insertion": RAMInsertionTrainConfig,
        "usb_pickup_insertion": USBPickupInsertionTrainConfig,
        "object_handover": ObjectHandoverTrainConfig,
        "egg_flip": EggFlipTrainConfig,
    })
