# ollama_utils.py
import os
import socket
import subprocess
import time
import signal
from contextlib import contextmanager
from typing import Optional

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434

_ollama_proc: Optional[subprocess.Popen] = None
_started_by_us: bool = False


def is_ollama_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_ollama_serve() -> None:
    """
    ollama serve를 백그라운드로 실행.
    이미 실행 중이면 아무것도 하지 않음.
    """
    global _ollama_proc, _started_by_us

    if is_ollama_running():
        _started_by_us = False
        return

    if os.name == "nt":
        # Windows: 새 프로세스 그룹으로 실행 (나중에 taskkill /T 로 트리 종료)
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_PROCESS_GROUP = 0x00000200

        _ollama_proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
        )
    else:
        # macOS/Linux: 새 세션(=프로세스 그룹)으로 실행해서 killpg 가능
        _ollama_proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # 핵심
        )

    _started_by_us = True
    # 디버그(원하면 지워도 됨)
    print(f"[ollama_utils] started ollama serve (pid={_ollama_proc.pid})")


def ensure_ollama_running(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    wait_seconds: float = 10.0,
    poll_interval: float = 0.2,
) -> None:
    if not is_ollama_running(host, port):
        start_ollama_serve()

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_ollama_running(host, port):
            return
        time.sleep(poll_interval)

    raise RuntimeError(
        f"Ollama 서버를 자동으로 시작했지만 {host}:{port} 포트가 열리지 않았습니다."
    )


def stop_ollama_if_started_by_us(grace_seconds: float = 2.0) -> None:
    """
    우리가 직접 띄운 ollama serve만 종료.
    Windows는 taskkill로 트리 종료, mac/linux는 killpg로 그룹 종료.
    """
    global _ollama_proc, _started_by_us

    if not _started_by_us:
        return

    if _ollama_proc is None:
        _started_by_us = False
        return

    pid = _ollama_proc.pid
    try:
        if os.name == "nt":
            # /T: 자식 포함, /F: 강제
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            # start_new_session=True로 실행했기 때문에 pid가 프로세스 그룹 리더가 됨
            try:
                os.killpg(pid, signal.SIGTERM)
                time.sleep(grace_seconds)
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        _ollama_proc = None
        _started_by_us = False
        print("[ollama_utils] stopped ollama serve (started by us)")


@contextmanager
def ollama_session(wait_seconds: float = 10.0):
    ensure_ollama_running(wait_seconds=wait_seconds)
    try:
        yield
    finally:
        stop_ollama_if_started_by_us()