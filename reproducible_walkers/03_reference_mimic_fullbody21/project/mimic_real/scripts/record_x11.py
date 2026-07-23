import argparse
import time

import imageio.v2 as imageio
import numpy as np
from PIL import ImageGrab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--display", default=":1")
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=18.0)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8, macro_block_size=None)
    start = time.perf_counter()
    next_frame = start
    try:
        while time.perf_counter() - start < args.seconds:
            img = ImageGrab.grab(bbox=(0, 0, 2560, 1440), xdisplay=args.display)
            img = img.resize((1280, 720))
            writer.append_data(np.asarray(img))
            next_frame += 1.0 / args.fps
            delay = next_frame - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

