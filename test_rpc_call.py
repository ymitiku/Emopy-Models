import jsonrpcclient
import base64
import cv2

with open('turtles.png', 'rb') as f:
    img = f.read()
    image_64 = base64.b64encode(img).decode('utf-8')

if __name__ == '__main__':
    jsonrpcclient.request("http://127.0.0.1:{}".format(8000), "classify",image=image_64, image_type='png')
