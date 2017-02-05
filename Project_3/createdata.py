
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import cv2
import numpy as np


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

    def batch_data(self, X_train, y_train, batch_size):
        images = np.zeros([batch_size, 64, 64, 3], dtype=np.float)
        steers = np.zeros(batch_size, dtype=np.float)
        steering_threshold = 0.1
        steering_prob = 0.8

        while True:
            i = 0
            for index in np.random.permutation(X_train.shape[0]):
                center, left, right = X_train[index]
                steering = y_train[index]

                image, steering = self.get_augument(center, left, right, steering)
                while abs(steering) < steering_threshold and np.random.rand() < steering_prob:
                    image, steering = self.get_augument(center, left, right, steering)

                images[i] = self.img_preprocess(image)
                steers[i] = steering
                i += 1
                if i == batch_size:
                    break
            yield images, steers

    def get_augument(self, center, left, right, steering):
        image, steering = self.select_image(center, left, right, steering)
        image, steering = self.random_flip(image, steering)
        image, steering = self.random_translate(image, steering, 100, 10)
        return image, steering

    def load_image(self, file):
        path = os.path.join(file.strip())
        image = mpimg.imread(path)

        return image

    def select_image(self, center, left, right, steering):
        choice = np.random.choice(3)
        if choice == 0:
            return self.load_image(left), steering + 0.2
        elif choice == 1:
            return self.load_image(right), steering - 0.2
        return self.load_image(center), steering

    def random_flip(self, image, steering):
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering = -steering
        return image, steering

    def random_translate(self, image, steering, range_x, range_y):
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering = steering + trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))

        return image, steering

    def img_preprocess(self, image):
        # Crop useless scene and focus on track
        img_crop = image[60:140, :, :]
        # Resize image
        img_resize = cv2.resize(img_crop, (64, 64), interpolation=cv2.INTER_AREA)
        # Convert to YUV
        image = cv2.cvtColor(img_resize, cv2.COLOR_RGB2YUV)

        return image