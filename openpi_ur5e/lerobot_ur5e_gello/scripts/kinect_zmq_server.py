"""Kinect v2 camera ZMQ server.

Streams Kinect v2 frames over ZMQ using the same protocol as lerobot_camera_zmq.

Run this in the conda 'skill' env where pylibfreenect2 works:

    conda activate skill
    pip install pyzmq  # if not already installed
    python scripts/kinect_zmq_server.py --port 5556 --topic kinect

Then in lerobot recording, use:
    --robot.type=ur5e_dual_cam

With the dual_cam config modified to use ZMQ for overview_cam:
    overview_cam: ZMQCameraConfig(tcp_address="tcp://localhost:5556", topic="kinect", ...)
"""

import argparse
import time
import base64
import json
import numpy as np
import cv2

try:
    import zmq
except ImportError:
    print("ERROR: zmq not installed. Run: pip install pyzmq")
    exit(1)


def start_kinect():
    """Initialize Kinect v2 via pylibfreenect2."""
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener
    from pylibfreenect2 import FrameType
    from pylibfreenect2 import createConsoleLogger, setGlobalLogger, LoggerLevel

    setGlobalLogger(createConsoleLogger(LoggerLevel.Warning))

    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        raise RuntimeError("No Kinect v2 found!")

    serial = fn.getDeviceSerialNumber(0)
    print(f"Opening Kinect v2 (serial: {serial})...")

    device = fn.openDevice(serial)
    listener = SyncMultiFrameListener(FrameType.Color)
    device.setColorFrameListener(listener)
    device.startStreams(rgb=True, depth=False)

    # Wait for stabilization
    for _ in range(10):
        frames = listener.waitForNewFrame()
        listener.release(frames)

    print(f"Kinect v2 ready (serial: {serial})")
    return fn, device, listener


def main():
    parser = argparse.ArgumentParser(description="Kinect v2 ZMQ frame server")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB port")
    parser.add_argument("--topic", type=str, default="kinect", help="ZMQ topic")
    parser.add_argument("--width", type=int, default=640, help="Output width")
    parser.add_argument("--height", type=int, default=480, help="Output height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--encoding", type=str, default="jpeg", choices=["jpeg", "raw"],
                       help="Frame encoding (jpeg=smaller bandwidth, raw=faster)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100")
    args = parser.parse_args()

    _fn, device, listener = start_kinect()

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{args.port}")
    print(f"ZMQ PUB server on tcp://*:{args.port} (topic='{args.topic}')")
    print(f"Output: {args.width}x{args.height} @ {args.fps}fps, encoding={args.encoding}")
    print("Press Ctrl+C to stop.\n")

    dt = 1.0 / args.fps
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            t0 = time.time()

            frames = listener.waitForNewFrame()
            color = frames["color"]

            # Kinect color: BGRX (1920x1080x4)
            img = color.asarray()[:, :, :3]  # Drop alpha -> BGR
            img = img[:, :, ::-1]  # BGR -> RGB
            listener.release(frames)

            # Resize
            if img.shape[0] != args.height or img.shape[1] != args.width:
                img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

            # Encode frame matching lerobot_camera_zmq protocol
            if args.encoding == "jpeg":
                # Convert RGB to BGR for JPEG encoding
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                _, jpeg_bytes = cv2.imencode('.jpg', bgr,
                                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                frame_b64 = base64.b64encode(jpeg_bytes.tobytes()).decode('ascii')
                data = {
                    "encoding": "jpeg",
                    "frame_bytes": frame_b64,
                    "shape": [args.height, args.width, 3],
                    "dtype": "uint8",
                }
            else:
                # Raw RGB bytes
                frame_b64 = base64.b64encode(img.tobytes()).decode('ascii')
                data = {
                    "encoding": "raw",
                    "frame_bytes": frame_b64,
                    "shape": [args.height, args.width, 3],
                    "dtype": "uint8",
                }

            # Send as [topic, json_data]
            topic_bytes = args.topic.encode('utf-8')
            data_bytes = json.dumps(data).encode('utf-8')
            socket.send_multipart([topic_bytes, data_bytes])

            frame_count += 1
            if frame_count % (args.fps * 10) == 0:
                elapsed = time.time() - t_start
                actual_fps = frame_count / elapsed
                print(f"  Frames: {frame_count} | Actual FPS: {actual_fps:.1f}")

            # Pace
            elapsed = time.time() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        print(f"\nStopped. Frames: {frame_count}, Duration: {elapsed:.1f}s, Avg FPS: {frame_count/elapsed:.1f}")
    finally:
        device.stop()
        device.close()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
