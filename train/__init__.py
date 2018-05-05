from models import getMultiInputEmopyModel
import keras

def start_training(args):
    """Builds and trains a emopy model
    
    Arguments:
        args {dict} -- dictionary containing model and training parameters
    """

    model = getMultiInputEmopyModel(args.input_shape,7)
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(args.lr),metrices=["accuracy"])
    model.summary()


    # train_model(model,args)
