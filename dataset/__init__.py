import os
from utils import EMOTIONS,EMOTION2INTEGER
import cv2
import numpy as np
import dlib
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.interpolation import rotate

# Dlib shape predictor to detect 68 face landmarks

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# datagenerator for generating augmented images
datagenerator = ImageDataGenerator(
                rotation_range = 20,
                width_shift_range = 0.1,
                height_shift_range = 0.1,
                zoom_range = 0.1,
                horizontal_flip=True,
                data_format="channels_last"
                 
            )
def load_dataset_files(dataset_dir):
    """Loads train and test image file names from given directory.
    
    Arguments:
        dataset_dir {str} -- directory which contains train and test folders
    
    Returns:
        list,list -- train and test dataset
    """

    train_image_files = []
    train_labels = []
    for emotion_folder in os.listdir(os.path.join(dataset_dir,"train")):
        for img_file in os.listdir(os.path.join(dataset_dir,"train",emotion_folder)):
            train_image_files+=[img_file]
            train_labels+=[EMOTION2INTEGER[emotion_folder]]
    



    test_image_files = []
    test_labels = []
    for emotion_folder in os.listdir(os.path.join(dataset_dir,"test")):
        for img_file in os.listdir(os.path.join(dataset_dir,"test",emotion_folder)):
            test_image_files+=[img_file]
            test_labels+=[EMOTION2INTEGER[emotion_folder]]
    

    return [train_image_files,train_labels],[test_image_files,test_labels]
def get_dlib_points(image):
    """Extracts 68 face landmark points from face image
    
    Arguments:
        image {np.ndarray} -- face image
    
    Returns:
        np.ndarray -- 68 face landmark points
    """

    output = np.zeros((68,2))
    shapes = predictor(image,dlib.rectangle(0,0,image.shape[1],image.shape[0]))
    for i in range(68):
        output[i] = [shapes.part(i).x,shapes.part(i).y]
    return output
def load_images_features(files_dir, image_files,labels,image_shape,augmentation=True):
    """loads face images and extracts features(landmark features) from face images.
    
    Arguments:
        files_dir {str} -- emotions folder directory
        image_files {list | np.ndarray} -- array containing image file names
        labels {int} -- emotion label for each image files
        image_shape {tuple | list} -- shape of output image files
    
    Keyword Arguments:
        augmentation {bool} -- if set true generates augmented images (default: {True})
    
    Returns:
        tuple -- images, dlib points, dlib points distances and angles from respective centroid, labels of each images
    """

    assert len(image_shape)==3,"Image shape should be length 3"
    output = np.zeros((len(image_files),image_shape[0],image_shape[1],image_shape[2]))
    dlib_points = np.zeros((len(image_files),68,2))
    dlib_points_distances = np.zeros((len(image_files),68,1))
    dlib_points_angles = np.zeros((len(image_files),68,1))

    
    
    for index in range(len(image_files)):   
        img = cv2.imread(os.path.join(files_dir,EMOTIONS[labels[index]],image_files[index]))
        if not(image_shape[0]==img.shape[0] and image_shape[1]==img.shape[1]):
            img = cv2.resize(img,(image_shape[0],image_shape[1]))
        if image_shape[2]==1:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img.astype(np.uint8)

        if augmentation:
            img = np.expand_dims(img,2)

            img = datagenerator.random_transform(img)
            img = np.squeeze(img)
        dpts = get_dlib_points(img)
       

        centroid = np.array([dpts.mean(axis=0)])
        dsts = distance_between(dpts,centroid)
        angles = angles_between(dpts,centroid)

        
        dsts = dsts.reshape(68,1)
        angles = angles.reshape(68,1)
        img = img.reshape(image_shape[0],image_shape[1],1)

        output[index] = img
        dlib_points[index] = dpts
        dlib_points_distances[index] = dsts
        dlib_points_angles[index] = angles

    dlib_points = np.expand_dims(dlib_points,1)
    dlib_points_distances = np.expand_dims(dlib_points_distances,1)
    dlib_points_angles = np.expand_dims(dlib_points_angles,1)

    return output,dlib_points,dlib_points_distances,dlib_points_angles,labels

def load_face_images(files_dir, image_files,labels,image_shape,augmentation=True):
    """loads face images
    
    Arguments:
        files_dir {str} -- emotions folder directory
        image_files {list | np.ndarray} -- array containing image file names
        labels {list | np.ndarray} -- emotion label for each image files
        image_shape {tuple | list} -- shape of output image files
    
    Keyword Arguments:
        augmentation {bool} -- if set true generates augmented images (default: {True})
    
    Returns:
        tuple -- images, dlib points, dlib points distances and angles from respective centroid, labels of each images
    """

    assert len(image_shape)==3,"Image shape should be length 3"
    output = np.zeros((len(image_files),image_shape[0],image_shape[1],image_shape[2]))
   

    
    
    for index in range(len(image_files)):   
        img = cv2.imread(os.path.join(files_dir,EMOTIONS[labels[index]],image_files[index]))
        if not(image_shape[0]==img.shape[0] and image_shape[1]==img.shape[1]):
            img = cv2.resize(img,(image_shape[0],image_shape[1]))
        if image_shape[2]==1:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img.astype(np.uint8)

        if augmentation:
            img = np.expand_dims(img,2)

            img = datagenerator.random_transform(img)
            img = np.squeeze(img)
        
        dpts = get_dlib_points(img)
     
        img = img.reshape(image_shape[0],image_shape[1],1)

        output[index] = img
       
    return output,labels
def distance_between(v1,v2):
    """Calculates euclidean distance between two vectors. 
    If one of the arguments is matrix then the output is calculated for each row
    of that matrix.

    Parameters
    ----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector
    
    Returns:
    --------
    numpy.ndarray
        Matrix if one of the arguments is matrix and vector if both arguments are vectors.
    """

    
    diff = v2 - v1
    diff_squared = np.square(diff)
    dist_squared = diff_squared.sum(axis=1) 
    dists = np.sqrt(dist_squared)
    return dists

def angles_between(v1,v2):
    """Calculates angle between two point vectors. 
    Parameters
    ----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector
    
    Returns:
    --------
    numpy.ndarray
        Vector if one of the arguments is matrix and scalar if both arguments are vectors.
    """
    dot_prod = (v1 * v2).sum(axis=1)
    v1_norm = np.linalg.norm(v1,axis=1)
    v2_norm = np.linalg.norm(v2,axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cosine_of_angle = np.divide(dot_prod,(v1_norm * v2_norm)).reshape(68,1)

    angles = np.arccos(np.clip(cosine_of_angle,-1,1))

    return angles
def generate_indexes(length,randomize=True):
    indexes = np.arange(length)
    if randomize:
        np.random.shuffle(indexes)
    return indexes
def compute_roll_angle(dlib_points):
    """Calculates face roll angle from dlib_points. 
    This method uses method specified here(http://www.scitepress.org/Papers/2015/53083/53083.pdf)
    to calculate roll angle.
    Arguments:
        files_dir {str} -- emotions folder directory
    
    Returns:
        float -- angle in degrees

    """
    left_eye_center = dlib_points[42:48].mean(axis=0)
    right_eye_center = dlib_points[36:42].mean(axis=0)

    dx = left_eye_center[0] - right_eye_center[0]
    dy = left_eye_center[1] - right_eye_center[1]

    radians = np.arctan(dy/dx)
    return np.degrees(radians)
    


def generator_face_features(dataset_dir,train_files,train_labels,args):
    if type(train_files) == list:
        train_files = np.array(train_files)
    if type(train_labels) == list:
        train_labels = np.array(train_labels)

    dataset_folder = os.path.join(dataset_dir,"train")
    while(True):
        indexes = generate_indexes(len(train_files))
        for i in range(len(train_files)//args.batch):
            current_indexes = indexes[i*args.batch:(i+1)*args.batch]

            current_files = train_files[current_indexes]
            current_labels = train_labels[current_indexes]
            
            images,dlib_points,dlib_points_distances,dlib_points_angles,labels = load_images_features(dataset_folder,current_files,current_labels,args.input_shape)

            images = images.astype(np.float32)/255
            IMAGE_HEIGHT = args.input_shape[0]
            dlib_points = dlib_points.astype(np.float32)/IMAGE_HEIGHT
            dlib_points_distances = dlib_points_distances.astype(np.float32)/IMAGE_HEIGHT
            dlib_points_angles = dlib_points_angles.astype(np.float32)/np.pi

            x = [images,dlib_points,dlib_points_distances,dlib_points_angles]
            y = np.eye(7)[labels]

            yield x,y

def generator_dlib_features(dataset_dir,train_files,train_labels,args):
    if type(train_files) == list:
        train_files = np.array(train_files)
    if type(train_labels) == list:
        train_labels = np.array(train_labels)

    dataset_folder = os.path.join(dataset_dir,"train")
    while(True):
        indexes = generate_indexes(len(train_files))
        for i in range(len(train_files)//args.batch):
            current_indexes = indexes[i*args.batch:(i+1)*args.batch]

            current_files = train_files[current_indexes]
            current_labels = train_labels[current_indexes]
            
            _,dlib_points,dlib_points_distances,dlib_points_angles,labels = load_images_features(dataset_folder,current_files,current_labels,args.input_shape)

            IMAGE_HEIGHT = args.input_shape[0]
            dlib_points = dlib_points.astype(np.float32)/IMAGE_HEIGHT
            dlib_points_distances = dlib_points_distances.astype(np.float32)/IMAGE_HEIGHT
            dlib_points_angles = dlib_points_angles.astype(np.float32)/np.pi

            x = [dlib_points,dlib_points_distances,dlib_points_angles]
            y = np.eye(7)[labels]

            yield x,y
def generator_face_images(dataset_dir,train_files,train_labels,args):

    if type(train_files) == list:
        train_files = np.array(train_files)
    if type(train_labels) == list:
        train_labels = np.array(train_labels)

    dataset_folder = os.path.join(dataset_dir,"train")
    while(True):
        indexes = generate_indexes(len(train_files))
        for i in range(len(train_files)//args.batch):
            current_indexes = indexes[i*args.batch:(i+1)*args.batch]

            current_files = train_files[current_indexes]
            current_labels = train_labels[current_indexes]
            
            images,labels = load_face_images(dataset_folder,current_files,current_labels,args.input_shape)

            images = images.astype(np.float32)/255
            
           

            x = images
            y = np.eye(7)[labels]

            yield x,y