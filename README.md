# Emopy-Models

Emotion recognition using facial expression demo.

## How to run demo
### Image demo

```
python -m demo --mtype np --ttype image --path /path-to-image/image-file 
```
Where ```--mtype``` is model type. It can be either np(neutral positive classifier) or ava(basic seven emotion classifier including neutral). ```--ttype``` is type of of demo either ```image``` , ```video``` or ```webcam```. ```--path``` is full path to image. 

### video demo
```
python -m demo --mtype np --ttype video --path /path-to-video/video-file 
```
### web demo
```
python -m demo --mtype np --ttype webcam 
```

### Dependancies

* tensorflow >= 1.0
* keras
* opencv >= 3.0
* dlib 
* numpy

* [shape_predictor_68_face_landmarks.dat][sp]

#### N.B

* **opencv should be compiled with ffmpeg support.**
* **Conda virtual environment can be created using the following command.**

 ```
 conda env create -f requirements.yml -n emopy_2
 ```
* **shape_predictor should be inside root directory of this project.**
 
 This will create virtual env with name ```emopy```.

 [sp]: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
