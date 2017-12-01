
from keras.models import model_from_json
from . import web_cam_demo,video_demo,image_demo
import argparse 
import os



# with open("/home/mtk/iCog/projects/emopy/models/all-neut-pos.json") as model_file:
#     model = model_from_json(model_file.read())
#     model.load_weights("/home/mtk/iCog/projects/emopy/models/all-neut-pos.h5")
#     # image_demo(model,"/home/mtk/iCog/projects/emopy/test-images/ang2.jpg")
    # web_cam_demo(model)
    # video_demo(model,"/home/mtk/iCog/projects/emopy/test-videos/1.mp4")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtype", default="neutral-positive")
    parser.add_argument("--ttype",default="webcam", type=str)
    parser.add_argument("--path",default=".", type=str)
    args = parser.parse_args()
    print args.mtype
    print args.ttype
    print args.path

    if not args.mtype  in ["np",'ava']:
        print "--mtype should be either np or ava"
        return
    if not args.ttype  in ["image","webcam",'video']:
        print "--mtype should be either image, webcam or video"
        return
    if not os.path.exists(args.path):
        print "invalid path for --path"
        return
    if args.ttype == "image":
        image_demo(args.mtype,args.path)
    elif args.ttype == "video":
        video_demo(args.mtype,args.path) 
    else:
        web_cam_demo(args.mtype);
    print args.ttype
        

if __name__ == "__main__":
    main()
    