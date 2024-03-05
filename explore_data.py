from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.io.psee_loader import PSEELoader

DATA_PATH = Path("./data/detection_dataset_duration_60s_ratio_1.0/train/")
TEMP_DAT = "17-03-30_12-53-58_1098500000_1158500000_td.dat"
HW_SIZE = (304, 240)

"""
    Get basic info of an event dat file
"""

tmp_vd = PSEELoader(str(DATA_PATH / TEMP_DAT))
tmp_vd.get_size()  # [240, 304]
tmp_vd.event_count()
tmp_vd.total_time()  # measured in micro sec (us)

"""
    Histogram of number of events per dat file (60s) on average
"""
evpersec = []
for fn in DATA_PATH.glob("*.dat"):
    print(fn)
    video = PSEELoader(str(fn))
    print(video.event_count(), video.total_time())
    evpersec.append(video.event_count() / video.total_time())

print(evpersec)

"""
    Histogram of sparsity per X ms
"""


def ev2buf(ev, buf):
    for (_, x, y, p) in ev:
        buf[x][y][p] += 1
    return buf


DELTA_T = 10_000  # 10 ms

sparsities = []
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
        sparsities.append(sparsity)
        buf.fill(0)
        print(sparsity)

plt.hist(sparsities, bins=100)
plt.savefig(f"sparsity_{DELTA_T/1000}ms.png")
ev1 = tmp_vd.load_delta_t(1000)  # load 1 ms
buf1 = ev2buf(ev1)
np.count_nonzero(buf1.sum(axis=-1)) / HW_SIZE[0] / HW_SIZE[1]

print(ev1, len(ev1))
ev2 = tmp_vd.load_delta_t(1000)
print(ev2, len(ev2))
