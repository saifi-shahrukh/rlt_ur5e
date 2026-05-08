# 01 — Full Environment Setup from Scratch

## Prerequisites

- **Ubuntu 22.04+** (tested on 24.04)
- **Python 3.10**
- **NVIDIA GPU** with CUDA 12.x drivers (driver 550+)
- **UR5e robot** connected via Ethernet
- **Robotiq Hand-E gripper** on UR Tool port
- **Intel RealSense D435** (wrist camera, USB 3.0)
- **Kinect Xbox One v2** (overview camera, USB 3.0)

## System Dependencies

```bash
# Basic build tools
sudo apt update
sudo apt install -y build-essential cmake git curl wget
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# For RealSense
sudo apt install -y libusb-1.0-0-dev libglfw3-dev

# For Kinect v2 (libfreenect2)
sudo apt install -y libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev
sudo apt install -y opencl-headers libva-dev libjpeg-dev
# Follow: https://github.com/OpenKinect/libfreenect2

# For pynput (keyboard control)
sudo apt install -y xdotool
```

## NVIDIA / CUDA Setup

```bash
# Verify NVIDIA driver
nvidia-smi  # Should show driver 550+ and CUDA 12.x

# If not installed:
sudo apt install -y nvidia-driver-550
sudo reboot
```

## Network Setup (Robot)

Configure a static IP on the Ethernet interface connected to the UR5e:

| Setting | Value |
|---------|-------|
| PC IP | 172.22.1.100 |
| Robot IP | 172.22.1.139 |
| Subnet | 255.255.255.0 |

Verify:
```bash
ping 172.22.1.139  # Should reply
```

## Kinect v2 udev Rules

```bash
sudo cp /path/to/libfreenect2/platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Clone & Install

See [02_uv_environment_setup.md](02_uv_environment_setup.md) for detailed Python environment setup.
