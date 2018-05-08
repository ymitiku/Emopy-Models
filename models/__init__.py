from keras.layers import Conv2D,MaxPool2D,Input,Dropout,Flatten,concatenate,Dense
from keras import backend as K
from keras.models import Model,model_from_json

def DlibAnglesLayer(dlib_points):
    pass
def DlibDistancesLayer(dlib_points):
    pass

def getImageInputModel(image_shape,num_class):
    image_input = Input(shape=image_shape)

    image_layer = Conv2D(32,[3,3],strides=[1,1],padding="same",activation="relu")(image_input)
    image_layer = Dropout(0.2)(image_layer)
    image_layer = Conv2D(32,[3,3],strides=[1,1],padding="same",activation="relu")(image_layer)
    image_layer = Dropout(0.2)(image_layer)
    image_layer = MaxPool2D(pool_size=(2,2))(image_layer)
    image_layer = Conv2D(64,[3,3],strides=[1,1],padding="same",activation="relu")(image_layer)
    image_layer = Dropout(0.2)(image_layer)
    image_layer = MaxPool2D(pool_size=(2,2))(image_layer)

    image_layer = Flatten()(image_layer)
    output = Dense(128,activation="relu")(image_layer)
    output = Dropout(0.2)(output)
    output = Dense(252,activation="relu")(output)
    output = Dropout(0.2)(output)
    output = Dense(num_class,activation="softmax")(output)

    model = Model(inputs=image_input,outputs=output)
    return model
def getDlibFeaturesInputModel(num_class):
    dlib_points_input = Input(shape=(1,68,2))
    dlib_points_layer = Conv2D(32,[1,3],strides=(1,1),padding="same",activation="relu")(dlib_points_input)
    dlib_points_layer = Conv2D(32,[1,3],strides=(1,1),padding="same",activation="relu")(dlib_points_layer)
    dlib_points_layer = MaxPool2D(pool_size=(1,2))(dlib_points_layer)
    dlib_points_layer = Conv2D(64,[1,3],strides=(1,1),padding="same",activation="relu")(dlib_points_layer)
    dlib_points_layer = MaxPool2D(pool_size=(1,2))(dlib_points_layer)
    dlib_points_layer = Flatten()(dlib_points_layer)


    dlib_points_distances_input = Input(shape=(1,68,1))

    dlib_points_distances_layer = Conv2D(32,[1,3],strides=(1,1),padding="same",activation="relu")(dlib_points_distances_input)
    dlib_points_distances_layer = Conv2D(32,[1,3],strides=(1,1),padding="same",activation="relu")(dlib_points_distances_layer)
    dlib_points_distances_layer = MaxPool2D(pool_size=(1,2))(dlib_points_distances_layer)
    dlib_points_distances_layer = Conv2D(64,[1,3],strides=(1,1),padding="same",activation="relu")(dlib_points_distances_layer)
    dlib_points_distances_layer = MaxPool2D(pool_size=(1,2))(dlib_points_distances_layer)
    dlib_points_distances_layer = Flatten()(dlib_points_distances_layer)

    dlib_points_angles_input = Input(shape=(1,68,1))

    dlib_points_angles_layer = Conv2D(32,[1,3],strides=(1,1),padding="same",activation="relu")(dlib_points_angles_input)
    dlib_points_angles_layer = Conv2D(32,[1,3],strides=(1,1),padding="same",activation="relu")(dlib_points_angles_layer)
    dlib_points_angles_layer = MaxPool2D(pool_size=(1,2))(dlib_points_angles_layer)
    dlib_points_angles_layer = Conv2D(64,[1,3],strides=(1,1),padding="same",activation="relu")(dlib_points_angles_layer)
    dlib_points_angles_layer = MaxPool2D(pool_size=(1,2))(dlib_points_angles_layer)
    dlib_points_angles_layer = Flatten()(dlib_points_angles_layer)
    
    merged = concatenate([dlib_points_layer,dlib_points_distances_layer,dlib_points_angles_layer])

    output = Dense(128,activation="relu")(merged)
    output = Dropout(0.2)(output)
    output = Dense(252,activation="relu")(output)
    output = Dropout(0.2)(output)
    output = Dense(num_class,activation="softmax")(output)

    model = Model(inputs=[dlib_points_input,dlib_points_distances_input,dlib_points_angles_input],outputs=output)
    return model
def getMultiInputEmopyModel(image_input_model,face_features_model,image_shape,num_class):
    """Gets model which will have three inputs(image, dlib key points,
    dlib key points distances and dlib key points angles from centroid.
    ) layers.
    
    Arguments:
        image_input_model {keras.model.Model} -- model with image input
        face_features_input_model {keras.model.Model} -- model with dlib face features input
        image_shape {list} -- input image shape
    Returns:
        keras.model.Model -- Four input model
    """

    image_model_last_layer = image_input_model.layers[9].output
    features_last_layer = face_features_model.layers[21].output
    merged = concatenate([image_model_last_layer,features_last_layer])


    output = Dense(128,activation="relu")(merged)
    output = Dropout(0.2)(output)
    output = Dense(252,activation="relu")(output)
    output = Dropout(0.2)(output)
    output = Dense(num_class,activation="softmax")(output)
    
   
    dlib_points_input,dlib_points_distances_input,dlib_points_angles_input  = face_features_model.inputs
    
    model = Model(inputs=(image_input_model.input,dlib_points_input,dlib_points_distances_input,dlib_points_angles_input),outputs=output)
    return model


