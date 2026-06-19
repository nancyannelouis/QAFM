"""
중복 실행 방지 락.

같은 스크립트(데이터셋별)가 이미 돌고 있는데 모르고 또 실행하면, GPU를 두
프로세스가 나눠 쓰며 둘 다 느려지고 결과 파일(JSON/체크포인트) 쓰기가 겹칠
위험이 있음. 스크립트 시작 시 락 파일을 확인해 이미 실행 중이면 즉시 종료.
"""

import os
import sys
import atexit

_LOCK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_locks")


def _is_pid_running(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _release_lock(lock_path: str) -> None:
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except OSError:
        pass


def acquire_lock(lock_name: str) -> None:
    """
    lock_name(보통 '{데이터셋}_{스크립트}') 기준으로 중복 실행 여부 확인.
    이미 같은 이름으로 실행 중인 프로세스가 살아있으면 경고 출력 후 즉시 종료.
    아니면 락 파일을 만들고, 정상/비정상 종료 시 자동 삭제되도록 등록.

    os.O_CREAT|O_EXCL로 "파일이 없을 때만 생성"을 원자적으로 수행 — 두
    프로세스가 거의 동시에 시작해도 OS 레벨에서 하나만 성공하도록 보장함
    (단순 "exists() 확인 후 쓰기"는 그 사이에 경쟁(race condition)이 생겨
    둘 다 통과해버릴 수 있음).
    """
    os.makedirs(_LOCK_DIR, exist_ok=True)
    lock_path = os.path.join(_LOCK_DIR, f"{lock_name}.lock")

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            with open(lock_path) as f:
                content = f.read().strip()
            if content.isdigit() and _is_pid_running(int(content)):
                print(f"[중복 실행 방지] '{lock_name}'은 이미 PID {content}로 실행 중입니다. "
                      f"중복 실행을 막기 위해 종료합니다.")
                print(f"  (정말 새로 실행하려면 {lock_path} 파일을 지우고 다시 실행하세요.)")
                sys.exit(1)
            # PID가 죽어있는 stale 락이면 지우고 재시도
            try:
                os.remove(lock_path)
            except OSError:
                pass
            continue
        else:
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            break

    atexit.register(_release_lock, lock_path)
