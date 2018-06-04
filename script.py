import base64
import random
import os
import numpy as np
from aiohttp import web
from jsonrpcserver.aio import methods
from jsonrpcserver.exceptions import InvalidParams
from demo import load_model_from_args, start_image_demo

'''
In this function, we would call the fucntion with args set as values
'''
class Args:
    def __init__(self):
        self.json = 'models/models/model-ff.json'
        self.weights = 'models/models/model-ff.h5'
        self.model_input = 'image'
        # TODO currenty we save, so this should be set this False for now
        self.snet = False
        self.gui = False
        self.path = ''
        self.image = ''
class Model:
    def __init__(self):
        self.args = Args()
        self.model = load_model_from_args(self.args)
    def predict(self):
        return start_image_demo(self.args, self.model)

app = web.Application()
model = Model()

@methods.add
async def classify(**kwargs):
    image = kwargs.get("image", None)
    image_type = kwargs.get("image_type", None)
    if image is None:
        raise InvalidParams("image is required")
    if image_type is None:
        raise InvalidParams("image type is required")
    binary_image = base64.b64decode(image)
    # this requires that we save the file. 
    current_files = os.listdir('tmp')
    tmp_file_name = 'tmp/tmp_' + str(random.randint(0,100000000000)) +'_.' + str(image_type)
    while tmp_file_name in current_files:
        tmp_file_name = 'tmp/tmp_' + str(random.randint(0,100000000000)) +'_.' + str(image_type)
    with open(tmp_file_name,'wb') as f:
        f.write(binary_image)
    model.args.path = 'tmp.'+str(image_type)
    bounding_boxes, emotions = model.predict()
    return {"bounding boxes": str(bounding_boxes),"predictions": str(emotions)}
async def handle(request):
    request = await request.text()
    response = await methods.dispatch(request)

    if response.is_notification:
        return web.Response()
    else:
        return web.json_response(response, status=response.http_status)

if __name__ == '__main__':
    app.router.add_post('/', handle)
    # create a tmp folder if it doesn't exist to hold files
    try: 
        os.mkdir('tmp')
    except OSError:
        print('tmp folder already exists, not created')
    web.run_app(app, host="127.0.0.1", port=8000)
