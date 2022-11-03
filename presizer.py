import argparse
import logging
import os
import os.path
import sys
from multiprocessing import Pool

# import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def resize_image(fpaths):
    # Ideally we would take 2 separate args but then we would need to
    # use Pool.starmap instead of Pool.imap and Pool.starmap does not
    # play well with tqdm.
    in_fpath, out_fpath = fpaths
    src = cv2.imread(in_fpath)
    if src is None:
        _logger.warn("Failed to cv2.imread '%s' so skipping it", in_fpath)
        return
    borderType = cv2.BORDER_REFLECT_101
    # get the h and w
    h, w = src.shape[:2]
    src = src[20 : h - 20, 20 : w - 20, :]
    if h > w:
        right = h - w
        image = cv2.copyMakeBorder(src, 0, 0, 0, right, borderType)
    elif w > h:
        top = w - h
        image = cv2.copyMakeBorder(src, top, 0, 0, 0, borderType)
    else:
        image = src

    # save image:
    cv2.imwrite(out_fpath, image)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-procs",
        type=int,
        help="Number of concurrent processes to use (images are split between the processes).  Defaults to Python's os.cpucount.",
    )
    parser.add_argument(
        "--in-dir",
        help="Directory to which file paths are relative to (defaults to current/working directory)",
    )
    parser.add_argument(
        "out_dir",
        help="Directory where to save resized images",
    )
    parser.add_argument(
        "in_filelist",
        help="File with one filepath (relative to IN-DIR) per line.",
    )
    args = parser.parse_args(argv[1:])

    in_dir = args.in_dir if args.in_dir else os.curdir

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    starmap_args = []
    required_out_dirs = set()
    with open(args.in_filelist, "r") as fh:
        for line in fh:
            rel_fpath = line.strip()
            in_fpath = os.path.join(in_dir, rel_fpath)
            out_fpath = os.path.join(out_dir, rel_fpath)
            required_out_dirs.add(os.path.dirname(out_fpath))
            starmap_args.append((in_fpath, out_fpath))

    # Build the directory structure on main process and ahead of time
    # so that we don't have to check it all the time in the workers
    # (and avoid possible race conditions).
    for required_out_dir in required_out_dirs:
        os.makedirs(required_out_dir, exist_ok=True)

    with Pool(args.n_procs) as pool:
        # We use imap instead of starmap because tqdm does not play
        # easily with starmap.
        for _ in tqdm(
            pool.imap_unordered(resize_image, starmap_args),
            total=len(starmap_args),
        ):
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
