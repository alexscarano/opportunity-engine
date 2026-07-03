import sys
from src.utils import time_s
from pathlib import Path

def log(namespace: str, data: str) -> None:
    path = Path(f"./data/log/{namespace}_{time_s()}.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")

class LogContext:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.log_data = []

    def write(self, message):
        try:
            self.stdout.write(message)
        except UnicodeEncodeError:
            enc = getattr(self.stdout, 'encoding', 'utf-8') or 'utf-8'
            self.stdout.write(message.encode(enc, errors='replace').decode(enc))
        self.log_data.append(message)

    def flush(self):
        self.stdout.flush()
        self.stderr.flush()

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if exc_type:
            import traceback
            tb_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self.write(tb_str)
        log(self.namespace, "".join(self.log_data))
