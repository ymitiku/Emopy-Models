import argparse
from train import start_training
import cv2
from skimage import feature
import numpy as np
import dlib
import tensorflow as tf 
import keras 

def get_cmd_args():
    """ Parse user command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_dir",default="dataset",type=str)
    parser.add_argument("-e","--epoch",default=10,type=int)
    parser.add_argument("-b","--batch",default=100,type=int)
    parser.add_argument("-s","--step",default=1000,type=int)
    parser.add_argument("-l","--lr",default=1e-4,type=float)
    parser.add_argument("-i","--input_shape",nargs=3,type=int,default=[48,48,1])
    parser.add_argument("-m","--model_output",type=str,default="model")
    parser.add_argument("-f","--features",type=str,default="all")


    args = parser.parse_args()
    return args

def main():
    """Start of training program.
    """
    np.random.seed(1)
    tf.set_random_seed(2)
    args = get_cmd_args()
    if args.input_shape[2]!=1:
        raise Exception("Currenly tested for only gray scale images. input_shape should be [height,width,1]")
    start_training(args)
   



if __name__ == '__main__':
    main()