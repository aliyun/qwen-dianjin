
import time
from pathlib import Path


class ProgressLogger:
    def __init__(self, base_path, shard, num_shards, total):
        base = Path(base_path)
        self.path = base.with_name(f"{base.stem}_shard{shard}{base.suffix}")
        self.shard = shard
        self.num_shards = num_shards
        self.total = total
        self.start = time.time()

    def _emit(self, done, correct, finished=False):
        elapsed = time.time() - self.start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (self.total - done) / rate if rate > 0 else 0
        acc = f"{correct/done:.4f}" if done else "0.0000"
        line = (
            f"shard={self.shard} done={done} total={self.total} "
            f"correct={correct} acc={acc} elapsed={elapsed:.0f}s "
            f"eta={eta:.0f}s{' DONE' if finished else ''}\n"
        )
        try:
            with open(self.path, "w") as f:
                f.write(line)
        except OSError:
            tmp = Path("/tmp") / self.path.name
            with open(tmp, "w") as f:
                f.write(line)

    def update(self, done, correct):
        self._emit(done, correct)

    def done(self, done, correct):
        self._emit(done, correct, finished=True)
