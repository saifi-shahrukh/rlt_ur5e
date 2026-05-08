import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from . import msgpack_numpy
from .base_policy import BasePolicy


class WebsocketClientPolicy(BasePolicy):
    """Policy wrapper that talks to a remote OpenPI policy server."""

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws://") or host.startswith("wss://"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info("Connecting to server at %s...", self._uri)

        use_ssl = self._uri.startswith("wss://")

        ssl_context = None
        if use_ssl:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                kwargs = dict(
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                )
                if ssl_context is not None:
                    kwargs["ssl"] = ssl_context

                conn = websockets.sync.client.connect(self._uri, **kwargs)
                metadata = msgpack_numpy.unpackb(conn.recv())
                logging.info("Connected to server successfully!")
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Server not ready, retrying in 5s...")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Connection error: {e}")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error from server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        return None

    def close(self) -> None:
        if hasattr(self, "_ws") and self._ws:
            self._ws.close()
