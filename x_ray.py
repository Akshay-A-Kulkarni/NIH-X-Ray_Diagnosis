import os
import cv2
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split


class preprocessor:
    '''
    Pre-processes image data into a 2D matrix format acceptable to most ML frameworks
    provided by Sci-Kit Learn.

    Also ensures that processed data is standardized by mean.
    '''
    def __init__(self, csv_path, image_path, img_width, img_height, labels = None):
        self.csv_path = csv_path
        self.image_path = image_path
        self.labels = labels
        self.img_width = img_width
        self.img_height = img_height

    def prepare_x_ray_data(self):

        # Load file from path and drop NA values
        x_ray_dev = pd.read_csv(self.csv_path, header = 0)
        x_ray_dev.dropna(inplace=True)

        # Remove rows with more than one Finding Label
        x_ray_dev = x_ray_dev[~x_ray_dev['Finding Labels'].str.contains('\\|')]

        # Create list of Finding Labels
        finding_Labels = sorted(x_ray_dev['Finding Labels'].unique())

        # Assign numerical Finding Labels to Finding IDs
        label_IDs = {}
        count = 1
        print("Assigning numerical Finding Labels to Finding IDs...")
        for i in finding_Labels:
            label_IDs[i] = count
            count = count + 1

        # Filter for only the selected input labels
        if self.labels is not None:
            x_ray_dev = x_ray_dev[x_ray_dev['Finding Labels'].isin(self.labels)]

        # Replace Finding Labels with Finding IDs
        x_ray_dev = x_ray_dev.replace({'Finding Labels': label_IDs})

        # Drop irrelevant/unused columns
        x_ray_dev = x_ray_dev.drop(x_ray_dev.columns[range(2, 11)], axis=1)

        # Put the Image Index and Finding ID in an iterable list
        result = [(x, y) for x, y in zip(x_ray_dev['Image Index'], x_ray_dev['Finding Labels'])]

        # Create a data frame of the flat image matrices and associated labels, and shuffle
        training_data = []
        
        print("Creating a data frame of the flat image matrics & associated labels and then shuffling...")
        for item in result:
            file = item[0]
            label = item[1]
            path = os.path.join(self.image_path, file)
            img = cv2.imread(path) # load the image as a matrix
            img = cv2.resize(img, (self.img_width, self.img_height))  # resize the image / matrix
            img = img.flatten()  # flatten the matrix
            data = np.append(img, label)  # combine the image matrix and label
            training_data.append(data)

        image_data = pd.DataFrame(training_data)#.sample(frac=1)

        X, y = image_data.iloc[:, :-1], image_data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1,stratify=y_train)

        scaler = preprocessing.Normalizer()
        X_train_s = scaler.fit_transform((X_train.values))
        X_train = pd.DataFrame(X_train_s, index=X_train.index, columns=X_train.columns)
        X_val_s = scaler.fit_transform((X_val.values))
        X_val = pd.DataFrame(X_val_s, index=X_val.index, columns=X_val.columns)
        X_test_s = scaler.fit_transform((X_test.values))
        X_test = pd.DataFrame(X_test_s, index=X_test.index, columns=X_test.columns)

        return X_train, X_val, X_test, y_train, y_val, y_test
