from pathlib import Path
import numpy as np
from src.io.psee_loader import PSEELoader

DATA_PATH = Path("./data/detection_dataset_duration_60s_ratio_1.0/train/")
HW_SIZE = (304, 240)
DELTA_T = 10_000  # 10 ms


def ev2buf(ev, buf):
    for _, x, y, p in ev:
        buf[x][y][p] += 1
    return buf


with open(f"sparsity_{DELTA_T/1000}ms.txt", "w") as f:
    buf = np.zeros((*HW_SIZE, 2), dtype=np.int32)
    for fn in DATA_PATH.glob("*.dat"):
        print(fn)
        video = PSEELoader(str(fn))
        while not video.done:
            events = video.load_delta_t(DELTA_T)
            buf = ev2buf(events, buf)
            sparsity = np.count_nonzero(buf.sum(axis=-1))
            sparsity /= HW_SIZE[0] * HW_SIZE[1]
            sparsity = 1 - sparsity
            buf.fill(0)
            f.write(f"{sparsity}\n")
