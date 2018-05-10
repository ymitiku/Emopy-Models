from keras.models import model_from_json
import cv2
import dlib
import numpy as np
from utils import EMOTIONS
from dataset import get_dlib_points,distance_between,angles_between
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

def load_model_from_args(args):
    with open(args.json) as json_file:
        model = model_from_json(json_file.read())
        model.load_weights(args.weights)
        return model


def overlay(frame, rectangles, text, color=(48, 12, 160)):
        """
        Draw rectangles and text over image
        Arguments:
            frame {numpy.ndarray} -- Image
        rectangles {list} -- Coordinates of rectangles to draw
        text {list} -- List of emotions to write
        color {tuple} -- Box and text color

        Returns:
        np.ndarray -- Most dominant emotion of each face drawn
        
        """

        for i, rectangle in enumerate(rectangles):
            cv2.rectangle(frame, (rectangle.left(),rectangle.top()), (rectangle.right(),rectangle.bottom()), color)
            cv2.putText(frame, text[i], (rectangle.left() + 10, rectangle.top() + 10), cv2.FONT_HERSHEY_DUPLEX, 0.4,color)
        return frame

def start_demo(args):
    model = load_model_from_args(args)
    if args.type=="image":
        start_image_demo(args,model)
    elif args.type=="video":
        start_video_demo(args,model)
    else:
        start_webcam_demo(args,model)
def get_max_index(predictions):
    max_value = predictions[0]
    max_index = 0
    for i in range(len(predictions)):
        if predictions[i] > max_value:
            max_index = i 
            max_value = predictions[i]
    return max_index,max_value
def get_emotion_str(predictions):
    max_index,confidence = get_max_index(predictions)
    return EMOTIONS[max_index],confidence


def get_emotion(features,model):
    predictions = model.predict(features)
    emotion,confidence = get_emotion_str(predictions[0])
    return emotion
def get_dlib_features(image):
    dpts = get_dlib_points(image)
    centroid = np.array([dpts.mean(axis=0)])
    dsts = distance_between(dpts,centroid)
    angles = angles_between(dpts,centroid)
    dsts = dsts.reshape(68,1)
    angles = angles.reshape(68,1)
    return dpts,dsts,angles

def start_image_demo(args,model):
    image_shape = model.inputs[0].shape.as_list()[1:]
    img = cv2.imread(args.path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_arrays = np.zeros((len(faces),image_shape[0],image_shape[1],image_shape[2]))
    face_rects = []
    emotions = []
    for i,face in enumerate(faces):
        face_imag = gray[
            max(0,face.top()):min(gray.shape[0],face.bottom()),
            max(0,face.left()):min(gray.shape[1],face.right())

        ]
        
        face_imag = cv2.resize(face_imag,(image_shape[0],image_shape[1]))
        dpts,dists,angles = get_dlib_features(face_imag)
        
        face_imag = face_imag.reshape(-1,image_shape[0],image_shape[1],image_shape[2])
        dists = dists.reshape(68,1)
        angles = angles.reshape(68,1)
        img = img.reshape(image_shape[0],image_shape[1],1)

        IMAGE_HEIGHT = image_shape[0]

        face_imag = face_imag.astype(np.float32)/255
       
        dpts = dpts.astype(np.float32)/IMAGE_HEIGHT
        dists = dists.astype(np.float32)/IMAGE_HEIGHT
        angles = angles.astype(np.float32)/np.pi

        e,_ = get_emotion([face_imag,dpts,dists,angles],model)
        emotions+=[e]
        face_rects+=[face]
    overlay(img,face_rects,emotions)
    cv2.imshow("Image Demo",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(cap,model):
    image_shape = model.inputs[0].shape.as_list()[1:]
    while cap.isOpened():
        _,frame = cap.read()

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        face_arrays = np.zeros((len(faces),image_shape[0],image_shape[1],image_shape[2]))
        face_rects = []
        emotions = []
        for i,face in enumerate(faces):
            face_imag = gray[
                max(0,face.top()):min(gray.shape[0],face.bottom()),
                max(0,face.left()):min(gray.shape[1],face.right())

            ]
            
            face_imag = cv2.resize(face_imag,(image_shape[0],image_shape[1]))

            dpts,dists,angles = get_dlib_features(face_imag)
            
            face_imag = face_imag.reshape(-1,image_shape[0],image_shape[1],image_shape[2])
            dists = dists.reshape(68,1)
            angles = angles.reshape(68,1)
            img = img.reshape(image_shape[0],image_shape[1],1)

            IMAGE_HEIGHT = image_shape[0]

            face_imag = face_imag.astype(np.float32)/255
        
            dpts = dpts.astype(np.float32)/IMAGE_HEIGHT
            dists = dists.astype(np.float32)/IMAGE_HEIGHT
            angles = angles.astype(np.float32)/np.pi

            e,_ = get_emotion([face_imag,dpts,dists,angles],model)
            emotions+=[e]
            face_rects+=[face]
        overlay(frame,face_rects,emotions)
        cv2.imshow("Video Demo",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def start_video_demo(args,model):
    cap = cv2.VideoCapture(args.path)
    process_video(cap,model)
def start_webcam_demo(args,model):
    cap = cv2.VideoCapture(-1)
    process_video(cap,model)