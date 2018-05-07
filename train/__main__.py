import argparse
from train import start_training
import cv2
from skimage import feature
import numpy as np
import dlib

def get_cmd_args():
    """ Parse user command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_dir",default="dataset",type=str)
    parser.add_argument("-e","--epoch",default=10,type=int)
    parser.add_argument("-b","--batch",default=100,type=int)
    parser.add_argument("-s","--step",default=1000,type=int)
    parser.add_argument("-l","--lr",default=1e-4,type=float)
    parser.add_argument("-i","--input_shape",type=int,default=[48,48,1])
    parser.add_argument("-m","--model_output",type=str,default="model")


    args = parser.parse_args()
    return args

def main():
    """Start of training program.
    """
    args = get_cmd_args()
    start_training(args)
   



if __name__ == '__main__':
    main()