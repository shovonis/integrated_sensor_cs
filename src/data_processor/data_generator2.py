import math
import tensorflow as tf
import numpy as np
import cv2
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""
1. Read meta data and get the head and eye 
2. Imbalance Class 

for each meta data:
    2. Merge head and eye with frame number
    3. Apply outlier detection and remove outlier  
    4. Time series window on the time_step 

:return 

1. All the normalized left frames in the clips 
2. All the normalized eye-tracking data 
3. All the normalized head-tracking data 

"""


class MultiModalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, meta_data, image_shape=(128, 128), time_step=5, batch_size=16, classification=True,
                 shuffle=True):
        self.batch_size = batch_size
        self.meta_data = meta_data
        self.shuffle = shuffle
        self.time_step = time_step
        self.classification = classification
        self.indexes = meta_data.index
        self.image_shape = image_shape
        self.classes = None
        self.tmp = []
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.meta_data) // self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_index = [i for i in range(index * self.batch_size, (index + 1) * self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.indexes[k] for k in batch_index]

        # Get eye and head
        eye = self.meta_data.loc[list_IDs_temp, ['eye']].to_numpy()
        eye = '/eye' + eye



        ead = self.prepare_time_series_data(data_path=head)
        eye = self.prepare_time_series_data(data_path=eye)
        # Target
        if self.classification:
            cs_class = self.meta_data.loc[list_IDs_temp, ['cs_class']].to_numpy()
            self.tmp.extend(cs_class)
            cs_class = tf.keras.utils.to_categorical(cs_class, num_classes=3)
        else:
            cs_class = self.meta_data.loc[list_IDs_temp, ['fms']].to_numpy()

        return [clips, optical, eye], cs_class

    def on_epoch_end(self):
        """Updates the indexes after each epoch"""
        self.classes = self.tmp
        self.indexes = np.arange(len(self.meta_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def frames_extraction(self, video_path):
        # Empty List declared to store video frames
        frames_list = []
        features = []
        for video in video_path:
            video = self.base_path + video
            video_reader = cv2.VideoCapture(video[0])
            while True:
                success, frame = video_reader.read()
                if not success:
                    break
                # Resize the Frame to fixed Dimensions and take only left image
                resized_frame = cv2.resize(frame[:, 0:256], self.image_shape)
                # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
                normalized_frame = resized_frame / 255
                # Appending the normalized frame into the frames list
                frames_list.append(normalized_frame)
            features.append(random.sample(frames_list, self.time_step_cnn))
            video_reader.release()

        return np.asarray(features)

    def prepare_time_series_data(self, paths):
        X = []

        for path in paths:
            current_eye_path = self.base_path + path
            # print(current_path)
            data = pd.read_csv(current_eye_path[0])
            data = data.drop(['#Frame'], axis=1)  # Drop the frame number
            scaler = StandardScaler()
            data = scaler.fit_transform(data)



            for i in range(data.shape[0]):
                end_ix = i + self.time_step_for_time_series
                if end_ix >= data.shape[0]:
                    break
                seq_X = data[i:end_ix]
                X.append(seq_X)





        sample_per_batch = random.sample(X, self.batch_size)
        return np.array(sample_per_batch)
