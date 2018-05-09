import argparse
from demo import start_demo

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j","--json",required=True,help="models json file path",type=str)
    parser.add_argument("-w","--weights",required=True,help="models weights file path",type=str)
    parser.add_argument("-i","--input",required=True,help="Type of input source. \nThis could be either image,webcam or video",type=str)
    parser.add_argument("-p","--path",type=str,help="path to input image file or video file.")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    start_demo(args)

if __name__ == '__main__':
    main()