"""Robotiq Hand-E gripper driver via UR tool communication (RS-485/Modbus).

Async interface using the Robotiq URCap socket on port 63352.
This is the same driver used in ur5e_serl, proven working with our hardware.
"""

import asyncio
import struct
import time
import enum
from typing import Optional


class HandEGripper:
    """Async Robotiq Hand-E gripper driver.

    Uses the Robotiq URCap socket server at <robot_ip>:63352.
    All commands are non-blocking (async/await).
    """

    class ObjectStatus(enum.IntEnum):
        MOVING = 0
        CONTACT_OPENING = 1
        CONTACT_CLOSING = 2
        AT_REQUESTED = 3

    def __init__(self, robot_ip: str, port: int = 63352):
        self.robot_ip = robot_ip
        self.port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False

    async def connect(self):
        """Connect to Robotiq URCap socket."""
        self._reader, self._writer = await asyncio.open_connection(
            self.robot_ip, self.port
        )
        self._connected = True
        # Read welcome message
        await asyncio.sleep(0.1)
        if self._reader.at_eof():
            raise RuntimeError(f"Hand-E connection failed at {self.robot_ip}:{self.port}")
        # Drain any welcome bytes
        try:
            await asyncio.wait_for(self._reader.read(1024), timeout=0.5)
        except asyncio.TimeoutError:
            pass

    async def activate(self):
        """Activate the gripper (required after power-on)."""
        await self._send_command("SET ACT 1\n")
        await asyncio.sleep(1.0)
        await self._send_command("SET GTO 1\n")
        await asyncio.sleep(0.5)

    async def _send_command(self, cmd: str):
        """Send a command string to the gripper."""
        if not self._connected:
            raise RuntimeError("Gripper not connected")
        self._writer.write(cmd.encode())
        await self._writer.drain()
        await asyncio.sleep(0.01)

    async def _get_var(self, var_name: str) -> int:
        """Get a variable value from the gripper."""
        cmd = f"GET {var_name}\n"
        self._writer.write(cmd.encode())
        await self._writer.drain()
        try:
            data = await asyncio.wait_for(self._reader.readline(), timeout=1.0)
            # Response format: "VAR_NAME value\n"
            parts = data.decode().strip().split()
            if len(parts) >= 2:
                return int(parts[-1])
        except (asyncio.TimeoutError, ValueError):
            pass
        return 0

    async def move(self, position: int, speed: int = 255, force: int = 150):
        """Move gripper to position (0=open, 255=closed)."""
        position = max(0, min(255, int(position)))
        speed = max(0, min(255, int(speed)))
        force = max(0, min(255, int(force)))
        await self._send_command(f"SET POS {position}\n")
        await self._send_command(f"SET SPE {speed}\n")
        await self._send_command(f"SET FOR {force}\n")
        await self._send_command("SET GTO 1\n")

    async def open(self, speed: int = 255, force: int = 100):
        """Fully open the gripper."""
        await self.move(0, speed, force)

    async def close(self, speed: int = 255, force: int = 150):
        """Fully close the gripper."""
        await self.move(255, speed, force)

    async def move_normalized(self, position: float, speed: int = 200, force: int = 120):
        """Move to normalized position (0.0=open, 1.0=closed)."""
        pos_int = int(np.clip(position, 0.0, 1.0) * 255)
        await self.move(pos_int, speed, force)

    async def get_position(self) -> int:
        """Get current position (0=open, 255=closed)."""
        return await self._get_var("POS")

    async def get_position_normalized(self) -> float:
        """Get current position normalized (0.0=open, 1.0=closed)."""
        pos = await self.get_position()
        return pos / 255.0

    async def get_object_status(self) -> 'HandEGripper.ObjectStatus':
        """Get object detection status."""
        val = await self._get_var("OBJ")
        try:
            return self.ObjectStatus(val)
        except ValueError:
            return self.ObjectStatus.MOVING

    async def disconnect(self):
        """Close the connection."""
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._connected = False


# Needed for move_normalized
import numpy as np
