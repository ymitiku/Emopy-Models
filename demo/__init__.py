
import dlib 
import cv2
import numpy as np
from constants import EMOTION_STATES,EMOTIONS
import os
from constants import THRESH_HOLD
from keras.models import model_from_json


detector = dlib.get_frontal_face_detector()



def get_dlib_shape_predictor():
    return dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


predictor = get_dlib_shape_predictor()

IMG_SIZE = (48,48)


def overlay(frame, rectangles, emotions, color=(48, 12, 160)):
       
        """
        Draw rectangles and emotion text over image

        Parameters
        ----------
        frame       : numpy.ndarray
            image on which rectangles  overlaid.
        rectnagles  : list
            face regions to overlay.
        emotions    : list
            emotions of each respective face
        color       : tupple
            color used to overlay rectangles and emotions text
        Returns
        -------
        numpy.ndarray
            image where rectangles and emotions are overlaid on it.
        """
        for i, rectangle in enumerate(rectangles):
            cv2.rectangle(frame, (rectangle.left(),rectangle.top()), (rectangle.right(),rectangle.bottom()), color)
            cv2.putText(frame, emotions[i], (rectangle.left() + 10, rectangle.top() + 10), cv2.FONT_HERSHEY_DUPLEX, 0.4,color)
        return frame
def sanitize(image):
    """
        Converts image into gray scale if it RGB image and resize it to IMG_SIZE

        Parameters
        ----------
        image : numpy.ndarray
        
        Returns
        -------
        numpy.ndarray
            gray scale image resized to IMG_SIZE
        """
    if image is None:
        return None
    assert len(image.shape) == 2 or len(image.shape) == 3,"Image dim should be either 2 or 3. It is "+str (len(image.shape))

    if len(image.shape) ==3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,IMG_SIZE)
    return image

def get_dlib_points(image,predictor):
    """
    Get dlib facial key points of face

    Parameters
    ----------
    image : numpy.ndarray
        face image.
    Returns
    -------
    numpy.ndarray
        68 facial key points
    """
    face = dlib.rectangle(0,0,image.shape[1]-1,image.shape[0]-1)
    img = image.reshape(IMG_SIZE[0],IMG_SIZE[1])
    shapes = predictor(img,face)
    parts = shapes.parts()
    output = np.zeros((68,2))
    for i,point in enumerate(parts):
        output[i]=[point.x,point.y]
    output = np.array(output).reshape((1,68,2))
    return output
def to_dlib_points(images,predictor):
    """
    Get dlib facial key points of faces

    Parameters
    ----------
    images : numpy.ndarray
        faces image.
    Returns
    -------
    numpy.ndarray
        68 facial key points for each faces
    """
    output = np.zeros((len(images),1,68,2))
    centroids = np.zeros((len(images),2))
    for i in range(len(images)):
        dlib_points = get_dlib_points(images[i],predictor)[0]
        centroid = np.mean(dlib_points,axis=0)
        centroids[i] = centroid
        output[i][0] = dlib_points
    return output,centroids
        
def get_distances_angles(all_dlib_points,centroids):
    """
    Get the distances for each dlib facial key points in face from centroid of the points and
    angles between the dlib points vector and centroid vector.

    Parameters
    ----------
    all_dlib_points : numpy.ndarray
        dlib facial key points for each face.
    centroid :
        centroid of dlib facial key point for each face
    Returns
    -------
    numpy.ndarray , numpy.ndarray
        Dlib landmarks distances and angles with respect to respective centroid.
    """
    all_distances = np.zeros((len(all_dlib_points),1,68,1))
    all_angles = np.zeros((len(all_dlib_points),1,68,1))
    for i in range(len(all_dlib_points)):
        dists = np.linalg.norm(centroids[i]-all_dlib_points[i][0],axis=1)
        angles = get_angles(all_dlib_points[i][0],centroids[i])
        all_distances[i][0] = dists.reshape(1,68,1)
        all_angles[i][0] = angles.reshape(1,68,1)
    return all_distances,all_angles
def angle_between(p1, p2):
    """
    Get clockwise angle between two vectors

    Parameters
    ----------
    p1 : numpy.ndarray
        first vector.
    p2 : numpy.ndarray
        second vector.
    Returns
    -------
    float
        angle in radiuns
    """
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang1 - ang2) % (2 * np.pi)
def get_angles(dlib_points,centroid):
    """
    Get clockwise angles between dlib landmarks of face and centroid of landmarks.

    Parameters
    ----------
    dlib_points : numpy.ndarray
        dlib landmarks of face.
    centroid : numpy.ndarray
        centroid of dlib landrmask.
    Returns
    -------
    numpy.ndarray
        dlib points clockwise angles in radiuns with respect to centroid vector
    """
    output = np.zeros((68))
    for i in range(68):
        angle = angle_between(dlib_points[i],centroid)
        output[i] = angle
    return output
def recognize(model,image,model_type):
    """
    Recognize emotion of each faces found in image. 

    Parameters
    ----------
    model : keras.models.Model
        model used to predict emotion.
    image : numpy.ndarray
        image which contains faces.
    Returns
    -------
    str, dlib.rectangle
        emotion and face rectangles.
    """
    faces  = detector(image)
    emotions = []
    rectangles = []
    for i in range(len(faces)):
        top = max(faces[i].top()-20,0)
        bottom = min(faces[i].bottom()+20,image.shape[0])
        left  = max(faces[i].left()-20,0)
        right = min(faces[i].right(),image.shape[1]+20)
        face = image[top:bottom, left:right]
        emotion,emotion_len = recognize_helper(model,face,model_type)
    
        if not emotion is None:
            if emotion_len == 2:
                emotions.append(EMOTION_STATES[emotion])
            else:
                emotions.append(EMOTIONS[emotion])
        else:
            emotions.append("unkown")
        rectangles.append(dlib.rectangle(left,top,right,bottom))
    return emotions,rectangles
def load_model(model_type):
    """
    Get model with specified 

    Parameters
    ----------
    model_type : str
       model type(either 'np' or 'ava')
    Returns
    -------
    keras.models.Model
        model for of type specified by model_type
    """
    with open(os.path.join("models",model_type+".json")) as model_file:
        model = model_from_json(model_file.read())
        model.load_weights(os.path.join("models",model_type+".h5"))
    return model
def arg_max(array):
    """
    Get index of maximum element of 1D array 

    Parameters
    ----------
    array : list
       
    Returns
    -------
    int
        index of maximum element of the array
    """
    max_value = array[0]
    max_index = 0
    for i,el in enumerate(array):
        if max_value< el:
            max_value=el
            max_index = i
    return max_index
def recognize_helper(model,face,model_type):
    """
    Recognize emotion single face image. 

    Parameters
    ----------
    model : keras.models.Model
        model used to predict emotion.
    image : numpy.ndarray
        face image.
    Returns
    -------
    str, int
        emotion and length of outputs of model.
    """
    face = sanitize(face)
    face = face.reshape(-1,48,48,1)
    if model_type!="ava-ii":
        dlibpoints,centroids = to_dlib_points(face,predictor)
        dists,angles = get_distances_angles(dlibpoints,centroids)
        dlibpoints = dlibpoints.astype(float)/50;
        dists = dists.astype(float)/50;
        angles = angles.astype(float)/50;
        face = face.reshape(face.shape[0], 48, 48, 1)
        face = face.astype('float32')
        face /= 255
        predictions = model.predict([face,dlibpoints,dists,angles])[0]
    else:
        predictions = model.predict(face)[0]
    emotion = arg_max(predictions)
    return emotion,len(predictions)
    

def image_demo(model_type,path):
    """
    demo to recognize emotions in still images. 

    Parameters
    ----------
    model_type : str
        model type(either 'np' or 'ava')
    path : str
        path to image.
    """
    model = load_model(model_type)
    img = cv2.imread(path)
    if img is None:
        raise Exception("Opencv failed to read image from given path.\n"+\
                    "check path '"+path+"' exists and valid image file.")
    emotions,rectangles = recognize(model,img,model_type)
    overlay(img,rectangles,emotions)
    cv2.imshow("Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(model, path,model_type,frame_width = 600):
    """
    demo to recognize emotions in videos. 

    Parameters
    ----------
    model : keras.models.Model
        model used to predict emotions in video
    path : str or int
        path is either path to video or -1 for webcam.
    """
    print "Path",path
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        FRAME_WIDTH = frame.shape[1]
        ratio = FRAME_WIDTH/float(frame_width)
        height = frame.shape[0]/float(ratio)
        frame = cv2.resize(frame,(frame_width,int(height)))
        
        emotions,rectangles = recognize(model,frame,model_type)
        overlay(frame,rectangles,emotions)
        cv2.imshow("Image",frame)
        if (cv2.waitKey(10) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()
def web_cam_demo(model_type,frame_width=600):
    """
    demo to recognize emotions using webcam. 

    Parameters
    ----------
    model_type : str
        model type(either 'np' or 'ava')
    """
    model = load_model(model_type)
    process_video(model,-1,model_type,frame_width)
def video_demo(model_type, path,frame_width=600):
    """
    demo to recognize emotions in videos. 

    Parameters
    ----------
    model_type : str
        model type(either 'np' or 'ava')
    path : str
        path to video.
    """
    model = load_model(model_type)
    process_video(model,path,model_type,frame_width)