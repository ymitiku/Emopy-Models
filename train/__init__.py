from models import getMultiInputEmopyModel
import keras
from dataset import load_dataset_files,load_images_features,generator
import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
def start_training(args):
    """Builds and trains a emopy model
    
    Arguments:
        args {dict} -- dictionary containing model and training parameters
    """

    model = getMultiInputEmopyModel(args.input_shape,7)
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(args.lr),metrics=["accuracy"])
    model.summary()
    train_model(model,args)
def train_model(model,args):
    train,test = load_dataset_files(args.dataset_dir)

    test_dataset_folder = os.path.join(args.dataset_dir,"test")
    test_images,test_dlib_points,test_dlib_points_distances,test_dlib_points_angles,test_labels = load_images_features(test_dataset_folder,test[0],test[1],(48,48,1))

    test_images = test_images.astype(np.float32)/255
    
    test_dlib_points = test_dlib_points.astype(np.float32)/48
    test_dlib_points_distances = test_dlib_points_distances.astype(np.float32)/48
    test_dlib_points_angles = test_dlib_points_angles.astype(np.float32)/np.pi

    x_test = [test_images,test_dlib_points,test_dlib_points_distances,test_dlib_points_angles]
    y_test = np.eye(7)[test_labels]

    model.fit_generator(generator(args.dataset_dir,train[0],train[1],args.batch),epochs=args.epoch,steps_per_epoch = args.step,verbose=1,validation_data=[x_test,y_test])
    model.save_weights("logs/models/model.h5")
    score = model.evaluate(x_test, y_test)
    with open("logs/logs/log.txt","a+") as log_file:
        log_file.write("Score:"+str(score)+"\n")
    
