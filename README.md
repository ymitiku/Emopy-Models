# Emotion recognition from face features
This project is aimed to train model that detects emotion from face image.

## How to preprocess datasets
This proejct uses [CK+ dataset](http://www.consortium.ri.cmu.edu/ckagree/) and  [Kaggle fer2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).  
The dataset should be saved inside single directory which contains ```train``` and ```test``` folders.
* To extract kaggle's dataset follow the instruction to download the dataset [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) 
  * Extract the dataset 
  * Inside this projects directory run the following code on terminal
```
python -m preprocess_fer -d /path-to-fer2013.csv-extracted 
```
* To process the ck+ dataset go to [this repository](https://github.com/mitiku1/Ck-dataset-preprocess)
* After preprocessing both dataset merge the two datasets mannually



### How to run training program
The four inputs model can be trained by three steps
* **shape_predictor should be inside root directory of this project. Shape predictor can be downloaded to project using the following script.**
```
cd /path-to-project
wget "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```
Training program can be run using the following command
```
python -m train [options]
```
Option to train program are
  * -d : Dataset directory path that contains train and test folders
  * -e : Number of epochs to train the model
  * -b : Training batch size
  * -s : Steps per epoch
  * -f : This option specifies the type of model to train. the options can be 'image', 'dlib' or 'all'. The default is 'all'. If user enters option that is not from ('image', 'dlib', 'all') then program continues with default option. Before training the 'all' model type the other two models should be trained and saved inside 'logs/models/' folder.
  * -l : learning rate
  * -i : image input shape


#### Step 1 - Training image input model
``` 
python -m train [options] -f image 
```
#### Step 3 - Training dlib features input model
``` 
python -m train [options] -f dlib 
```
#### Step 3 - Training the main model
``` 
python -m train [options] 
```

### Running Demo program
``` 
python -m demo [options]
```  
Options for demo program are
* -j : model json file
* -w : model weights
* -i : Face image source. This could be either `image`, `video` or `webcam`. Defualt is webcam.
* -p : Path to source file. If options -i is webcam this is not necessary.

## Requirements
* keras >= 2.0
* tensorflow >= 1.5
* dlib
* opencv >= 3.1
* sklearn
* numpy 
* pandas

# LICENSE
MIT License

Copyright (c) 2018 Mitiku Yohannes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 [sp]: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2