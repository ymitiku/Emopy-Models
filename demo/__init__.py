
import dlib 
import cv2
import numpy as np
from constants import EMOTION_STATES,EMOTIONS
import os
from constants import THRESH_HOLD
from keras.models import model_from_json


detector = dlib.get_frontal_face_detector()

MODEL_TYPE = 'pos-neut'


def get_dlib_shape_predictor():
    return dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


predictor = get_dlib_shape_predictor()

IMG_SIZE = (48,48)


def overlay(frame, rectangles, emotions, color=(48, 12, 160)):
        """
        Draw rectangles and emotion text over image

        :param Mat frame: Image
        :param list rectangles: Coordinates of rectangles to draw
        :param list emotions: List of emotions to write
        :param tuple color: Box and text color
        :return: Most dominant emotion of each face
        :rtype: list
        """
        for i, rectangle in enumerate(rectangles):
            cv2.rectangle(frame, (rectangle.left(),rectangle.top()), (rectangle.right(),rectangle.bottom()), color)
            cv2.putText(frame, emotions[i], (rectangle.left() + 10, rectangle.top() + 10), cv2.FONT_HERSHEY_DUPLEX, 0.4,color)
        return frame
def sanitize(img):
    if img is None:
        return None
    assert len(img.shape) == 2 or len(img.shape) == 3,"Image dim should be either 2 or 3. It is "+str (len(img.shape))

    if len(img.shape) ==3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,IMG_SIZE)
    return img

def get_dlib_points(image,predictor):
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
    output = np.zeros((len(images),1,68,2))
    centroids = np.zeros((len(images),2))
    for i in range(len(images)):
        dlib_points = get_dlib_points(images[i],predictor)[0]
        centroid = np.mean(dlib_points,axis=0)
        centroids[i] = centroid
        output[i][0] = dlib_points
    return output,centroids
        
def get_distances_angles(all_dlib_points,centroids):
    all_distances = np.zeros((len(all_dlib_points),1,68,1))
    all_angles = np.zeros((len(all_dlib_points),1,68,1))
    for i in range(len(all_dlib_points)):
        dists = np.linalg.norm(centroids[i]-all_dlib_points[i][0],axis=1)
        angles = get_angles(all_dlib_points[i][0],centroids[i])
        all_distances[i][0] = dists.reshape(1,68,1)
        all_angles[i][0] = angles.reshape(1,68,1)
    return all_distances,all_angles
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang1 - ang2) % (2 * np.pi)
def get_angles(dlib_points,centroid):
    output = np.zeros((68))
    for i in range(68):
        angle = angle_between(dlib_points[i],centroid)
        output[i] = angle
    return output
def recognize(model,image):
    faces  = detector(image)
    emotions = []
    rectangles = []
    for i in range(len(faces)):
        top = max(faces[i].top()-20,0)
        bottom = min(faces[i].bottom()+20,image.shape[0])
        left  = max(faces[i].left()-20,0)
        right = min(faces[i].right(),image.shape[1]+20)
        face = image[top:bottom, left:right]
        emotion,emotion_len = recognize_helper(model,face)
    
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
    with open(os.path.join("models",model_type+".json")) as model_file:
        model = model_from_json(model_file.read())
        model.load_weights(os.path.join("models",model_type+".h5"))
    return model
def arg_max(array):
    max_value = array[0]
    max_index = 0
    for i,el in enumerate(array):
        if max_value< el:
            max_value=el
            max_index = i
    return max_index
def recognize_helper(model,face):
    face = sanitize(face)
    face = face.reshape(-1,48,48,1)

    dlibpoints,centroids = to_dlib_points(face,predictor)
    dists,angles = get_distances_angles(dlibpoints,centroids)
    dlibpoints = dlibpoints.astype(float)/50;
    dists = dists.astype(float)/50;
    angles = angles.astype(float)/50;
    face = face.reshape(face.shape[0], 48, 48, 1)
    face = face.astype('float32')
    face /= 255
    predictions = model.predict([face,dlibpoints,dists,angles])[0]
    emotion = arg_max(predictions)
    if predictions[emotion]>THRESH_HOLD:
        return emotion,len(predictions)
    

def image_demo(model_type,path):
    model = load_model(model_type)
    img = cv2.imread(path)
    if img is None:
        raise Exception("Opencv failed to read image from given path.\n"+\
                    "check path '"+path+"' exists and valid image file.")
    emotions,rectangles = recognize(model,img)
    overlay(img,rectangles,emotions)
    cv2.imshow("Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(model, path):
    print "Path",path
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    ret,frame = cap.read()
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame,(300,240))
        emotions,rectangles = recognize(model,frame)
        overlay(frame,rectangles,emotions)
        cv2.imshow("Image",frame)
        if (cv2.waitKey(10) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()
def web_cam_demo(model_type):
    model = load_model(model_type)
    process_video(model,-1)
def video_demo(model_type, path):
    model = load_model(model_type)
    process_video(model,path)