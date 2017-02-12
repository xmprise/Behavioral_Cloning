import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import cv2
import numpy as np


def load_image(file):
    path = os.path.join(file.strip())
    image = mpimg.imread(path)

    return image


# Moves the image in the X and Y directions to create a virtual image.
# train better recovery by moving the axes along the angle and changing the angle.
def random_translate(image, steering, range_x, range_y):
    x_translation = range_x * (np.random.rand() - 0.5)
    y_translation = range_y * (np.random.rand() - 0.5)
    steering += x_translation * 0.005
    trans_model = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_model, (width, height))

    return image, steering


# Select the image and adjust the associated handle angle.
def select_image(center, left, right, steering):
    off_center = 0.2
    choice = np.random.choice(3)

    if choice == 0:
        return load_image(left), steering + off_center
    elif choice == 1:
        return load_image(right), steering - off_center
    return load_image(center), steering


# Use this to determine whether to flip the image horizontally
# and to reduce the bias toward one side.
def random_flip(image, steering):
    if np.random.rand() < 0.5:
        steering = -steering
        image = cv2.flip(image, 1)

    return image, steering


def img_preprocess(image):
    # Crop useless scene and focus on track and resize
    img_resize = cv2.resize(image[60:140, :], (64, 64))
    # Convert to color space
    hsv = cv2.cvtColor(img_resize, cv2.COLOR_RGB2HSV)
    # random brightness
    rand = np.random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand * hsv[:, :, 2]
    # back to origin color space
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return new_img


def get_augument(center, left, right, steering):

    image, steering = select_image(center, left, right, steering)
    image, steering = random_flip(image, steering)
    image, steering = random_translate(image, steering, 100, 10)

    return image, steering


class CreateData:

    def __init__(self, path):
        self.path = path

    # Read driving image and csv file path
    #csv_path = "./savedata/driving_log.csv"
    def load_data(self):
        csv_df = pd.read_csv(self.path)
        # Center, Left, Right, Steer, Throttle, Break, Speed
        img_df = csv_df[[0, 1, 2]].values
        steer = csv_df[[3]].values

        X_train, X_valid, y_train, y_valid = train_test_split(img_df, steer, test_size=0.2, random_state=0)

        return X_train, X_valid, y_train, y_valid

    # Create a training image that provides associated with the steering
    def batch_data(self, X_train, y_train, batch_size):
        images = np.zeros([batch_size, 64, 64, 3], dtype=np.float)
        steer = np.zeros(batch_size, dtype=np.float)
        steering_threshold = 0.1
        steering_prob = 0.8

        while True:
            i = 0
            for index in np.random.permutation(X_train.shape[0]):
                center, left, right = X_train[index]
                steering = y_train[index]

                image, steering = get_augument(center, left, right, steering)
                # Adjust the steering angle so that it has a non-zero steering angle
                # because the steering angle of the provided data is zero.
                while abs(steering) < steering_threshold and np.random.rand() < steering_prob:
                    image, steering = get_augument(center, left, right, steering)

                images[i] = img_preprocess(image)
                steer[i] = steering
                i += 1
                if i == batch_size:
                    break

            yield images, steer