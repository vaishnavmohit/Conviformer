import os
import numpy as np
from os.path import isfile, isdir
from tqdm import tqdm
# import tensorflow as tf
from glob import glob
import json
import pandas as pd
from os.path import normpath, basename
import math
from pathlib import Path
import argparse
import cv2
import sys

def main():
    
    # print command line arguments
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    for args in sys.argv[1:]:
        print(args)

    # reading files:
    TRAIN_DIR = "/cifs/data/tserre_lrs/projects/prj_fossils/data/raw_data/Herbarium_2021_FGVC8/train"
    SAVE_DIR = "/users/mvaishn1/scratch/datasets//Herbarium_2021_FGVC8_resize/train"
    dataframe = pd.read_csv('herb21in22.csv')


    for c, f in tqdm(enumerate(dataframe[['image_dir']].values)):
        if c >= start:
            # read image
            src = cv2.imread(os.path.join(TRAIN_DIR,f[0]))       
            borderType = cv2.BORDER_REFLECT_101
            # get the h and w
            h,w = src.shape[:2]
            src = src[20:h-20,20: w-20, :]
            save = os.path.join(SAVE_DIR,f[0])

            # if not Path(save).is_file():
            basepath,_ = os.path.split(save)
            Path(basepath).mkdir(parents=True, exist_ok=True)
            if h > w:
                right = h-w
                image = cv2.copyMakeBorder(src, 0, 0, 0, right, borderType)
            elif w > h:
                top = w-h
                image = cv2.copyMakeBorder(src, top, 0, 0, 0, borderType)
            else:
                image = src
            
            # save image:
            cv2.imwrite(save, image)
        if c == end:
            break

if __name__ == "__main__":
    main()