import math
import tensorflow as tf
import numpy as np
import cv2
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""Generate multimodal data from the data source"""


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, base_path, image_shape=(128, 128), time_step_image=100,
                 batch_size=16, time_step_time_series=300, classification=False, shuffle=True):
        """Initialization"""
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.classification = classification
        self.base_path = base_path
        self.indexes = dataset.index
        self.image_shape = image_shape
        self.classes = None
        self.tmp = []
        self.time_step_cnn = time_step_image
        self.time_step_for_time_series = time_step_time_series
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.dataset) // self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_index = [i for i in range(index * self.batch_size, (index + 1) * self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.indexes[k] for k in batch_index]
        # Clips
        clips = self.dataset.loc[list_IDs_temp, ['video_clip']].to_numpy()  # .reshape(-1)
        clips = '/clips' + clips
        clips = self.frames_extraction(video_path=clips)
        # # optical
        optical = self.dataset.loc[list_IDs_temp, ['optical']].to_numpy()
        optical = '/optic' + optical
        optical = self.frames_extraction(video_path=optical)
        # # Disparity
        disparity = self.dataset.loc[list_IDs_temp, ['disparity']].to_numpy()
        disparity = '/disp' + disparity
        disparity = self.frames_extraction(video_path=disparity)
        # Process Eye
        eye = self.dataset.loc[list_IDs_temp, ['eye']].to_numpy()
        eye = '/eye' + eye
        eye = self.prepare_time_series_data(data_path=eye)
        # Process Head
        head = self.dataset.loc[list_IDs_temp, ['head']].to_numpy()
        head = '/head' + head
        head = self.prepare_time_series_data(data_path=head)
        # Target
        if self.classification:
            cs_class = self.dataset.loc[list_IDs_temp, ['cs_class']].to_numpy()
            self.tmp.extend(cs_class)
            cs_class = tf.keras.utils.to_categorical(cs_class, num_classes=3)
        else:
            cs_class = self.dataset.loc[list_IDs_temp, ['fms']].to_numpy()

        return [clips, optical, eye], cs_class

    def on_epoch_end(self):
        """Updates the indexes after each epoch"""
        self.classes = self.tmp
        self.indexes = np.arange(len(self.dataset))
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

    def prepare_time_series_data(self, data_path=None):
        X = []
        for path in data_path:
            current_path = self.base_path + path
            # print(current_path)
            data = pd.read_csv(current_path[0])
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
