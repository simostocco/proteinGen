import shutil
from ..contracts import d_path


def require_budget(root, peak_bytes, reserve_bytes=10 * 1024**3):
    free = shutil.disk_usage(d_path(root)).free
    if free < reserve_bytes + peak_bytes:
        raise RuntimeError("BLOCKED_RESOURCE")
    return free
