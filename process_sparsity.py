from pathlib import Path
import sys
import argparse
import numpy as np
import sparse
from src.io.psee_loader import PSEELoader


def ev2buf(ev, buf):
    for _, x, y, p in ev:
        buf[x][y][p] += 1
    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-t", "--deltaT", dest="dt", default=10, type=int, help="window size in ms")
    parser.add_argument("-o", "--output", default="output", help="output folder")
    args = parser.parse_args()

    # check args
    if args.dt < 2 or args.dt > 32:
        print(f"{args.dt} too small or large", file=sys.stderr)
        # sparse COO int16 may overflow beyound designed dt range
        exit(1)

    if not Path(args.filename).exists():
        print(f"input file {args.filename} does not exist.", file=sys.stderr)
        exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"processing {Path(args.filename)}")

    # parse events to buffered video
    vd = PSEELoader(str(Path(args.filename)))
    dt = 1000 * args.dt 
    vd_len = np.ceil(vd.total_time() / dt).astype(np.int32)
    HW_SIZE = (304, 240)
    buf = np.zeros((*HW_SIZE, 2), dtype=np.int16)
    vd_buf = np.zeros((vd_len, *HW_SIZE, 2), dtype=np.int16)
    print(vd_buf.shape)
    idx = 0
    while not vd.done:
        evt = vd.load_delta_t(dt)
        buf = ev2buf(evt, buf)
        vd_buf[idx] = buf
        buf.fill(0)
        idx += 1

    # convert to sparse encoding and save
    vd_sp = sparse.COO(vd_buf)
    save_file = str(Path(args.output) / Path(args.filename).stem)+".npz"
    sparse.save_npz(save_file, vd_sp)
    print(f"finished {save_file}")
    
    # to load npz
    # vd_sp = sparse.load_npz(str(save_file))
    # vd = vd_sp.todense()


if __name__ == "__main__":
    main()
