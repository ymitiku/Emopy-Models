import os
from utils import EMOTIONS,EMOTION2INTEGER
import cv2
import numpy as np

def load_dataset_files(dataset_dir):
    train_image_files = []
    train_labels = []
    for emotion_folder in os.listdir(os.path.join(dataset_dir,"train")):
        for img_file in os.listdir(os.path.join(dataset_dir,"train",emotion_folder)):
            train_image_files+=[img_file]
            train_labels+=[EMOTION2INTEGER[emotion_folder]]
    train_indexes = range(len(train_image_files))
    np.random.shuffle(train_indexes)

    train_image_files = train_image_files[train_indexes]
    train_labels = train_labels[train_indexes]



    test_image_files = []
    test_labels = []
    for emotion_folder in os.listdir(os.path.join(dataset_dir,"test")):
        for img_file in os.listdir(os.path.join(dataset_dir,"test",emotion_folder)):
            test_image_files+=[img_file]
            test_labels+=[EMOTION2INTEGER[emotion_folder]]
    test_indexes = range(len(test_image_files))
    np.random.shuffle(test_indexes)

    test_image_files = test_image_files[test_indexes]
    test_labels = test_labels[test_indexes]

    return [train_image_files,train_labels],[test_image_files,test_labels]
def load_images(files_dir, image_files,labels,image_shape):
    assert len(image_shape)==3,"Image shape should be length 3"
    output = np.zeros((len(image_files),image_shape[0],image_shape[1],image_shape[2]))

    for index in range(len(image_files)):   
        img = cv2.imread(os.path.join(files_dir,image_files[index]))
        if not(image_shape[0]==img.shape[0] and image_shape[1]==img.shape[1]):
            img = cv2.resize(img,(image_shape[0],image_shape[1]))
        if image_shape[2]==1:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = img.reshape(image_shape[0],image_shape[1],1)
        output[index] = img
    return output
        