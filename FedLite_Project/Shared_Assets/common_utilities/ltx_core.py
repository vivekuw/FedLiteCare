"""Minimal localhost chunked file transfer helpers for LTX-style simulation."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from queue import Queue
from typing import Any, Callable

from FedLite_Project.Shared_Assets.common_utilities.common_utils import ensure_directory


def _read_json_line(connection: socket.socket) -> dict[str, Any]:
    buffer = bytearray()
    while True:
        chunk = connection.recv(1)
        if not chunk:
            raise ConnectionError("Connection closed before a complete JSON header was received.")
        if chunk == b"\n":
            break
        buffer.extend(chunk)
    return json.loads(buffer.decode("utf-8"))


def send_file_chunked(
    source_path: Path,
    target_host: str,
    target_port: int,
    chunk_size: int,
    socket_timeout_seconds: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Send a file to a localhost receiver using fixed-size chunks."""
    source_path = source_path.resolve()
    file_size = source_path.stat().st_size
    header = {
        "file_name": source_path.name,
        "file_size": file_size,
        **metadata,
    }

    bytes_sent = 0
    with socket.create_connection((target_host, target_port), timeout=socket_timeout_seconds) as connection:
        connection.settimeout(socket_timeout_seconds)
        connection.sendall(json.dumps(header).encode("utf-8") + b"\n")

        with source_path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                connection.sendall(chunk)
                bytes_sent += len(chunk)

        acknowledgement = _read_json_line(connection)

    return {
        "source_path": source_path,
        "target_host": target_host,
        "target_port": target_port,
        "file_size": file_size,
        "bytes_sent": bytes_sent,
        "acknowledgement": acknowledgement,
    }


def receive_file_chunked(
    bind_host: str,
    bind_port: int,
    destination_path: Path,
    chunk_size: int,
    socket_timeout_seconds: float,
    ready_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Receive a single file over localhost and save it in chunks."""
    destination_path = destination_path.resolve()
    ensure_directory(destination_path.parent)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind_host, bind_port))
        listener.listen(1)
        listener.settimeout(socket_timeout_seconds)
        if ready_event is not None:
            ready_event.set()

        connection, sender_address = listener.accept()
        with connection:
            connection.settimeout(socket_timeout_seconds)
            header = _read_json_line(connection)
            expected_size = int(header["file_size"])
            bytes_received = 0

            with destination_path.open("wb") as handle:
                while bytes_received < expected_size:
                    chunk = connection.recv(min(chunk_size, expected_size - bytes_received))
                    if not chunk:
                        raise ConnectionError("Connection closed before the complete file was received.")
                    handle.write(chunk)
                    bytes_received += len(chunk)

            acknowledgement = {
                "status": "ok",
                "received_bytes": bytes_received,
                "destination_path": str(destination_path),
            }
            connection.sendall(json.dumps(acknowledgement).encode("utf-8") + b"\n")

    return {
        "destination_path": destination_path,
        "bind_host": bind_host,
        "bind_port": bind_port,
        "header": header,
        "bytes_received": bytes_received,
        "sender_address": sender_address[0],
        "sender_port": sender_address[1],
    }


def start_receiver_thread(
    receiver_callable: Callable[..., dict[str, Any]],
    **receiver_kwargs: Any,
) -> tuple[threading.Thread, Queue[Any]]:
    """Run a blocking receive call in a background thread and wait for it to start listening."""
    result_queue: Queue[Any] = Queue(maxsize=1)
    ready_event = threading.Event()

    def _runner() -> None:
        try:
            result_queue.put(
                ("ok", receiver_callable(ready_event=ready_event, **receiver_kwargs))
            )
        except Exception as error:
            ready_event.set()
            result_queue.put(("error", error))

    receiver_thread = threading.Thread(target=_runner, daemon=True)
    receiver_thread.start()
    if not ready_event.wait(timeout=5):
        raise TimeoutError("Timed out while waiting for the LTX receiver to start.")
    return receiver_thread, result_queue


def finish_receiver_thread(receiver_thread: threading.Thread, result_queue: Queue[Any]) -> dict[str, Any]:
    """Wait for a receiver thread and surface any transfer error."""
    receiver_thread.join()
    status, payload = result_queue.get()
    if status != "ok":
        raise payload
    return payload
