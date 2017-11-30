
from keras.models import model_from_json
from . import web_cam_demo,video_demo,image_demo

with open("/home/mtk/iCog/projects/emopy/models/all-neut-pos.json") as model_file:
    model = model_from_json(model_file.read())
    model.load_weights("/home/mtk/iCog/projects/emopy/models/all-neut-pos.h5")
    # image_demo(model,"/home/mtk/iCog/projects/emopy/test-images/ang2.jpg")
    web_cam_demo(model)
    # video_demo(model,"/home/mtk/iCog/projects/emopy/test-videos/1.mp4")