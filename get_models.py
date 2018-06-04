import requests
import bz2
import math
import os
from tqdm import tqdm

model = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

response = requests.get(model, stream=True)
total_size = int(response.headers.get('content-length', 0))
print(total_size)
decompressor = bz2.BZ2Decompressor()
with open('shape_predictor_68_face_landmarks.dat', 'wb') as f:
    for data in tqdm(response.iter_content(1024), total=math.ceil(total_size//1024), unit='KB'):
        f.write(decompressor.decompress(data))
if ('shape_predictor_68_face_landmarks.dat' in os.listdir()):
    # TODO use checksum
    print('Model persumably downloaded')
else:
    print('Model isnot available')
