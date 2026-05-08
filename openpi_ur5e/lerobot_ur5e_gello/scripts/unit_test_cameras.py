#!/usr/bin/env python3
"""Unit test for both cameras in the UR5e + OpenPI pipeline.

Tests:
  1. RealSense D435 (wrist cam) via LeRobot interface
  2. Kinect v2 (overhead cam) via LeRobot interface
  3. Simultaneous dual-cam capture
  4. FPS / performance at 30Hz
  5. Robot config loading

Run:
  cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello
  source .venv/bin/activate
  python scripts/unit_test_cameras.py
"""

import sys
import os
import time
import warnings
import numpy as np

# Suppress warnings for clean output
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
os.environ.setdefault("LD_LIBRARY_PATH", f"/home/robolab-2/freenect2/lib:{os.environ.get('LD_LIBRARY_PATH', '')}")

# Suppress libfreenect2 [Info] logs early
try:
    from pylibfreenect2 import createConsoleLogger, setGlobalLogger, LoggerLevel
    setGlobalLogger(createConsoleLogger(LoggerLevel.Error))
except ImportError:
    pass

results = []

def run_test(name, func):
    """Run a test and track results."""
    print(f"\n{'─'*60}")
    print(f"  TEST: {name}")
    print(f"{'─'*60}")
    try:
        func()
        results.append((name, "PASS", ""))
        print(f"  ✓ PASSED")
    except Exception as e:
        results.append((name, "FAIL", str(e)))
        print(f"  ✗ FAILED: {e}")


# ===== TEST 1: Imports =====
def test_imports():
    import pyrealsense2 as rs
    print(f"    pyrealsense2: OK")
    
    from pylibfreenect2 import Freenect2
    print(f"    pylibfreenect2: OK")
    
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    print(f"    lerobot RealSenseCamera: OK")
    
    from lerobot_camera_kinect import KinectCameraConfig, KinectCamera
    print(f"    lerobot KinectCamera: OK")
    
    from lerobot_robot_ur5e import UR5EConfig, UR5EDualCamConfig
    print(f"    lerobot UR5EConfig: OK")
    
    import cv2
    print(f"    opencv: {cv2.__version__}")
    
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    print(f"    openpi_client: OK")


# ===== TEST 2: RealSense Detection =====
def test_realsense_detection():
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    assert len(devices) > 0, "No RealSense devices found!"
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"    Found: {name} (Serial: {serial})")


# ===== TEST 3: Kinect Detection =====
def test_kinect_detection():
    from pylibfreenect2 import Freenect2
    fn = Freenect2()
    n = fn.enumerateDevices()
    assert n > 0, "No Kinect v2 devices found!"
    for i in range(n):
        serial = fn.getDeviceSerialNumber(i)
        if isinstance(serial, bytes):
            serial = serial.decode()
        print(f"    Found: Kinect v2 (Serial: {serial})")


# ===== TEST 4: RealSense LeRobot Interface =====
def test_realsense_lerobot():
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot.cameras.configs import ColorMode
    
    config = RealSenseCameraConfig(
        serial_number_or_name="034422070605",
        fps=30, width=640, height=480,
        color_mode=ColorMode.RGB,
    )
    cam = RealSenseCamera(config)
    cam.connect(warmup=True)
    
    try:
        frame = cam.read()
        assert frame is not None, "Got None frame"
        assert frame.shape == (480, 640, 3), f"Wrong shape: {frame.shape}"
        assert frame.dtype == np.uint8, f"Wrong dtype: {frame.dtype}"
        assert frame.max() > 0, "Frame is all black"
        print(f"    Sync read: {frame.shape}, range=[{frame.min()}, {frame.max()}]")
        
        frame_async = cam.async_read(timeout_ms=2000)
        assert frame_async.shape == (480, 640, 3)
        print(f"    Async read: {frame_async.shape}")
    finally:
        cam.disconnect()


# ===== TEST 5: Kinect LeRobot Interface =====
def test_kinect_lerobot():
    from lerobot_camera_kinect import KinectCameraConfig, KinectCamera
    from lerobot.cameras.configs import ColorMode
    
    config = KinectCameraConfig(
        serial="000631452147",
        fps=30, width=640, height=480,
        color_mode=ColorMode.RGB,
        flip_horizontal=True,
    )
    cam = KinectCamera(config)
    cam.connect(warmup=True)
    
    try:
        frame = cam.read()
        assert frame is not None, "Got None frame"
        assert frame.shape == (480, 640, 3), f"Wrong shape: {frame.shape}"
        assert frame.dtype == np.uint8, f"Wrong dtype: {frame.dtype}"
        assert frame.max() > 0, "Frame is all black"
        print(f"    Sync read: {frame.shape}, range=[{frame.min()}, {frame.max()}]")
        
        frame_async = cam.async_read(timeout_ms=2000)
        assert frame_async.shape == (480, 640, 3)
        print(f"    Async read: {frame_async.shape}")
    finally:
        cam.disconnect()


# ===== TEST 6: Dual Camera Simultaneous =====
def test_dual_camera_simultaneous():
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot_camera_kinect import KinectCameraConfig, KinectCamera
    from lerobot.cameras.configs import ColorMode
    
    rs_cam = RealSenseCamera(RealSenseCameraConfig(
        serial_number_or_name="034422070605",
        fps=30, width=640, height=480, color_mode=ColorMode.RGB,
    ))
    k_cam = KinectCamera(KinectCameraConfig(
        serial="000631452147",
        fps=30, width=640, height=480, color_mode=ColorMode.RGB, flip_horizontal=True,
    ))
    
    rs_cam.connect(warmup=True)
    k_cam.connect(warmup=True)
    
    try:
        success = 0
        for i in range(10):
            t0 = time.time()
            rs_frame = rs_cam.read()
            k_frame = k_cam.read()
            dt = (time.time() - t0) * 1000
            if rs_frame is not None and k_frame is not None:
                success += 1
            if i < 3:
                print(f"    [{i}] wrist={rs_frame.shape} overhead={k_frame.shape} | {dt:.1f}ms")
        
        assert success >= 9, f"Only {success}/10 simultaneous captures succeeded"
        print(f"    Success: {success}/10 pairs captured")
    finally:
        rs_cam.disconnect()
        k_cam.disconnect()


# ===== TEST 7: FPS Performance =====
def test_fps_performance():
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot_camera_kinect import KinectCameraConfig, KinectCamera
    from lerobot.cameras.configs import ColorMode
    
    rs_cam = RealSenseCamera(RealSenseCameraConfig(
        serial_number_or_name="034422070605",
        fps=30, width=640, height=480, color_mode=ColorMode.RGB,
    ))
    k_cam = KinectCamera(KinectCameraConfig(
        serial="000631452147",
        fps=30, width=640, height=480, color_mode=ColorMode.RGB, flip_horizontal=True,
    ))
    
    rs_cam.connect(warmup=True)
    k_cam.connect(warmup=True)
    
    try:
        n_frames = 30
        t0 = time.time()
        for _ in range(n_frames):
            rs_cam.read()
            k_cam.read()
        elapsed = time.time() - t0
        fps = n_frames / elapsed
        
        print(f"    Dual-cam FPS: {fps:.1f} (target: 30)")
        print(f"    Avg frame pair time: {(elapsed/n_frames)*1000:.1f}ms")
        assert fps > 20, f"FPS too low: {fps:.1f} (need >20 for 30Hz with headroom)"
    finally:
        rs_cam.disconnect()
        k_cam.disconnect()


# ===== TEST 8: Robot Config =====
def test_robot_config():
    from lerobot_robot_ur5e import UR5EConfig, UR5EDualCamConfig
    
    sc = UR5EConfig()
    assert sc.ip == "172.22.1.139"
    assert "wrist_cam" in sc.cameras
    assert len(sc.cameras) == 1
    print(f"    UR5EConfig: ip={sc.ip}, cameras={list(sc.cameras.keys())}")
    
    dc = UR5EDualCamConfig()
    assert dc.ip == "172.22.1.139"
    assert "wrist_cam" in dc.cameras
    assert "overview_cam" in dc.cameras
    assert len(dc.cameras) == 2
    print(f"    UR5EDualCamConfig: ip={dc.ip}, cameras={list(dc.cameras.keys())}")


# ===== TEST 9: Image Resize for Model =====
def test_image_resize():
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot_camera_kinect import KinectCameraConfig, KinectCamera
    from lerobot.cameras.configs import ColorMode
    import cv2
    
    rs_cam = RealSenseCamera(RealSenseCameraConfig(
        serial_number_or_name="034422070605",
        fps=30, width=640, height=480, color_mode=ColorMode.RGB,
    ))
    k_cam = KinectCamera(KinectCameraConfig(
        serial="000631452147",
        fps=30, width=640, height=480, color_mode=ColorMode.RGB, flip_horizontal=True,
    ))
    
    rs_cam.connect(warmup=True)
    k_cam.connect(warmup=True)
    
    try:
        rs_frame = rs_cam.read()  # (480, 640, 3)
        k_frame = k_cam.read()   # (480, 640, 3)
        
        # Model expects 224x224
        rs_224 = cv2.resize(rs_frame, (224, 224))
        k_224 = cv2.resize(k_frame, (224, 224))
        
        assert rs_224.shape == (224, 224, 3)
        assert k_224.shape == (224, 224, 3)
        assert rs_224.dtype == np.uint8
        assert k_224.dtype == np.uint8
        print(f"    RealSense resized: {rs_frame.shape} -> {rs_224.shape}")
        print(f"    Kinect resized: {k_frame.shape} -> {k_224.shape}")
    finally:
        rs_cam.disconnect()
        k_cam.disconnect()


# ===== MAIN =====
def main():
    print("=" * 60)
    print("  UR5e + OpenPI Pipeline - Camera Unit Tests")
    print("=" * 60)
    
    tests = [
        ("1. Python imports", test_imports),
        ("2. RealSense D435 detection", test_realsense_detection),
        ("3. Kinect v2 detection", test_kinect_detection),
        ("4. RealSense LeRobot interface", test_realsense_lerobot),
        ("5. Kinect LeRobot interface", test_kinect_lerobot),
        ("6. Dual camera simultaneous capture", test_dual_camera_simultaneous),
        ("7. FPS performance (30Hz target)", test_fps_performance),
        ("8. Robot config loading", test_robot_config),
        ("9. Image resize for model (224x224)", test_image_resize),
    ]
    
    for name, func in tests:
        run_test(name, func)
    
    # Summary
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    
    for name, status, err in results:
        icon = "✓" if status == "PASS" else "✗"
        line = f"  {icon} {name}"
        if err:
            line += f" — {err}"
        print(line)
    
    print(f"\n  Results: {passed} passed, {failed} failed, {len(results)} total")
    print(f"{'='*60}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Cameras are ready for data collection.")
    else:
        print("\n⚠️  Some tests failed. Check hardware connections.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
