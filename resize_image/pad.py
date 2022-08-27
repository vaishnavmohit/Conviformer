import os
import numpy as np
from os.path import isfile, isdir
from tqdm import tqdm
# import tensorflow as tf
from glob import glob
from os.path import normpath, basename
import math
from pathlib import Path
import argparse
import cv2
import sys

def main():
    
    # print command line arguments
    for args in sys.argv[1:]:
        print(args)

    #setting training parameters
    
    #Load our dataset
    train_dir = '/users/mvaishn1/scratch/datasets/herbarium-2022-fgvc9/train_images/*/*/*'
    test_dir = '/users/mvaishn1/scratch/datasets/herbarium-2022-fgvc9/test_images/*/*'
    
    # list of images: 
    train_list = glob(train_dir)
    test_list = glob(test_dir)

    for f in tqdm(train_list+test_list):
        # read image
        src = cv2.imread(f)        
        borderType = cv2.BORDER_REFLECT_101
        # get the h and w
        h,w = src.shape[:2]
        src = src[20:h-20,20: w-20, :]
        save = f.replace('herbarium-2022-fgvc9', 'herbarium-2022-fgvc9_resize')
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

if __name__ == "__main__":
    main()