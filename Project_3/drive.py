import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import cv2
from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


def img_preprocess(image):
    # Crop useless scene and focus on track
    img_crop = image[60:140, :, :]
    # Resize image
    img_resize = cv2.resize(img_crop, (64, 64), interpolation=cv2.INTER_AREA)
    # Convert to YUV
    image = cv2.cvtColor(img_resize, cv2.COLOR_RGB2YUV)

    return image

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    img = np.asarray(image)
    img = img_preprocess(img)
    image_array = np.asarray(img)
    transformed_image_array = image_array[None, :, :, :]
    print(img.shape)
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle * 1., throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition h5. Model should be on the same path.')
    args = parser.parse_args()

    model = load_model(args.model)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
