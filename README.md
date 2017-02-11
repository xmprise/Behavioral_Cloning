#**Behavioral Cloning**

This project is Udacity [Self-Driving Car Nanodegree] (https://www.udacity.com/drive). The goal of the project is to complete all the track provided by the simulator, and the code written by it satisfies this. The structure of the used DNN was VGG16, and the acquired data was used the training mode of the simulator. For the simulator provided by Udacity, see [section](https://github.com/udacity/self-driving-car-sim). The data used here was obtained directly using the ps4 controller.

The model runs as shown below.

Track 1                       |  Track 2
:----------------------------:|:------------------------------:
![Track 1](imgfile/track1.gif) | ![Track 2](imgfile/track2.gif)
|[YouTube Link](https://youtu.be/j5FRHlkMdpM)|[YouTube Link](https://youtu.be/MDbgBqUcmnE)|

###Dependencies

This project used requires **Python 3.5** and using below libraries: 

- [TensorFlow](http://tensorflow.org)
- [Keras](https://keras.io/)
- [pandas](http://pandas.pydata.org/)
- [NumPy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [opencv](http://opencv.org/)

Drive.py only need below libraries:

- [flask-socketio](https://flask-socketio.readthedocs.io/en/latest/)
- [eventlet](http://eventlet.net/)
- [pillow](https://python-pillow.org/)

###Quick Start

- Install python libraries:

You need a [anaconda](https://www.continuum.io/downloads)
Next we need to install Tensorflow and Keras.

```Tensorflow
$ pip install tensorflow
if use GPU:
$ pip install tensorflow-gpu
```
```Keras
$ sudo pip install keras
```
```opencv
$ conda install -c https://conda.binstar.org/menpo opencv
```

- Learning the model
```python
python model.py
```

- Run the trained model
```python
python drive.py modelxxx.h5
