"""Minimal localhost chunked file transfer helpers for LTX-style simulation."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import socket
import ssl
import tempfile
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable

from FedLite_Project.Shared_Assets.common_utilities.common_utils import ensure_directory


def _calculate_sha256(path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with path.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


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
    use_compression: bool = True,
    ssl_context: ssl.SSLContext | None = None,
) -> dict[str, Any]:
    """Send a file to a localhost receiver using fixed-size chunks, with optional compression, checksum, and SSL."""
    source_path = source_path.resolve()
    original_size = source_path.stat().st_size
    sha256_checksum = _calculate_sha256(source_path)

    # Prepare for compression if enabled
    transfer_path = source_path
    temp_compressed_path = None
    if use_compression:
        temp_handle = tempfile.NamedTemporaryFile(
            suffix=source_path.suffix + ".gz.tmp",
            delete=False,
        )
        temp_handle.close()
        temp_compressed_path = Path(temp_handle.name)
        with source_path.open("rb") as f_in:
            with gzip.open(temp_compressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        transfer_path = temp_compressed_path

    transfer_size = transfer_path.stat().st_size
    header = {
        "file_name": source_path.name,
        "original_size": original_size,
        "transfer_size": transfer_size,
        "sha256": sha256_checksum,
        "compressed": use_compression,
        "use_ssl": ssl_context is not None,
        **metadata,
    }

    try:
        bytes_sent = 0
        raw_connection = socket.create_connection((target_host, target_port), timeout=socket_timeout_seconds)
        with raw_connection:
            if ssl_context:
                connection = ssl_context.wrap_socket(raw_connection, server_hostname=target_host)
            else:
                connection = raw_connection

            connection.settimeout(socket_timeout_seconds)
            connection.sendall(json.dumps(header).encode("utf-8") + b"\n")

            with transfer_path.open("rb") as handle:
                while True:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        break
                    connection.sendall(chunk)
                    bytes_sent += len(chunk)

            acknowledgement = _read_json_line(connection)
    finally:
        if temp_compressed_path and temp_compressed_path.exists():
            temp_compressed_path.unlink()

    return {
        "source_path": source_path,
        "target_host": target_host,
        "target_port": target_port,
        "original_size": original_size,
        "transfer_size": transfer_size,
        "file_size": transfer_size,
        "bytes_sent": bytes_sent,
        "sha256": sha256_checksum,
        "compressed": use_compression,
        "use_ssl": ssl_context is not None,
        "acknowledgement": acknowledgement,
    }


def receive_file_chunked(
    bind_host: str,
    bind_port: int,
    destination_path: Path,
    chunk_size: int,
    socket_timeout_seconds: float,
    ready_event: threading.Event | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> dict[str, Any]:
    """Receive a single file over localhost, handling optional decompression, checksum, and SSL."""
    destination_path = destination_path.resolve()
    ensure_directory(destination_path.parent)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind_host, bind_port))
        listener.listen(1)
        listener.settimeout(socket_timeout_seconds)
        if ready_event is not None:
            ready_event.set()

        raw_connection, sender_address = listener.accept()
        with raw_connection:
            if ssl_context:
                connection = ssl_context.wrap_socket(raw_connection, server_side=True)
            else:
                connection = raw_connection

            connection.settimeout(socket_timeout_seconds)
            header = _read_json_line(connection)
            transfer_size = int(header["transfer_size"])
            is_compressed = bool(header.get("compressed", False))
            expected_sha256 = header.get("sha256")
            
            bytes_received = 0
            temp_handle = tempfile.NamedTemporaryFile(
                suffix=destination_path.suffix + ".part",
                delete=False,
            )
            temp_handle.close()
            temp_path = Path(temp_handle.name)

            try:
                with temp_path.open("wb") as handle:
                    while bytes_received < transfer_size:
                        chunk = connection.recv(min(chunk_size, transfer_size - bytes_received))
                        if not chunk:
                            raise ConnectionError("Connection closed before the complete file was received.")
                        handle.write(chunk)
                        bytes_received += len(chunk)

                # Decompress if needed
                if is_compressed:
                    with gzip.open(temp_path, "rb") as f_in:
                        with destination_path.open("wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    temp_path.unlink()
                else:
                    if destination_path.exists():
                        destination_path.unlink()
                    shutil.move(str(temp_path), str(destination_path))
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            # Verify checksum
            actual_sha256 = _calculate_sha256(destination_path)
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ValueError(f"SHA-256 mismatch! Expected {expected_sha256}, got {actual_sha256}")

            acknowledgement = {
                "status": "ok",
                "received_bytes": bytes_received,
                "sha256_verified": True,
                "destination_path": str(destination_path),
            }
            connection.sendall(json.dumps(acknowledgement).encode("utf-8") + b"\n")

    return {
        "destination_path": destination_path,
        "bind_host": bind_host,
        "bind_port": bind_port,
        "header": header,
        "bytes_received": bytes_received,
        "sha256": actual_sha256,
        "compressed": is_compressed,
        "use_ssl": ssl_context is not None,
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
    try:
        status, payload = result_queue.get_nowait()
    except Empty:
        return receiver_thread, result_queue

    receiver_thread.join(timeout=0.1)
    if status != "ok":
        raise payload
    result_queue.put((status, payload))
    return receiver_thread, result_queue


def finish_receiver_thread(receiver_thread: threading.Thread, result_queue: Queue[Any]) -> dict[str, Any]:
    """Wait for a receiver thread and surface any transfer error."""
    receiver_thread.join()
    status, payload = result_queue.get()
    if status != "ok":
        raise payload
    return payload
