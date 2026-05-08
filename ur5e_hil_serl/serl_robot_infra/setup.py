from setuptools import setup, find_packages

setup(
    name="serl_robot_infra",
    version="0.1.0",
    description="UR5e + Hand-E robot infrastructure for HIL-SERL",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "pyrealsense2",
        "opencv-python",
        "pyspacemouse",
        "hidapi",
        "pyyaml",
        "scipy",
        "numpy",
        "ur-rtde",
        "pynput",
    ],
    extras_require={
        "kinect": ["pylibfreenect2"],
    },
    python_requires=">=3.9",
)
