# Copyright 2018 Northwest University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from keras.datasets import cifar10
from keras.utils import np_utils
from util import util

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)


class Generator:
    def __init__(self, opt):
        self.opt = opt
        self.datagen = image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
            # rotation_range=15,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True
        )
        self.dict = util.create_dict(opt.cap_scheme)
        print('Loading train and validation data...')
        if opt.isTune:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.data_load('real')
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.data_load('synthetic')

    def text2vec(self, text):
        text_len = len(text)
        # print(text)
        if text_len > self.opt.cap_len:
            raise ValueError('The max length of this captcha is {}' .format(self.opt.cap_len))
        if self.opt.char_set_len != len(self.dict):
            raise ValueError('The number of characters does not match to the dict')
        vector = np.zeros(self.opt.cap_len * self.opt.char_set_len)

        def char2pos(c):
            k = -1
            for (key, value) in self.dict.items():
                if value == c:
                    k = key
                    return k
            if k == -1:
                raise ValueError('Wrong with dict or text')
        for i, c in enumerate(text):
            idx = i * self.opt.char_set_len + char2pos(c)
            vector[idx] = 1
        return vector

    # load training and validation data
    def data_load(self, label):
        data_path = os.path.join(self.opt.dataroot, self.opt.cap_scheme, label)
        self.num_train_samples =min(self.opt.train_size, len(os.listdir(os.path.join(data_path, 'train'))))
        # num_test_samples = len(os.listdir(os.path.join(data_path, 'test')))
        self.num_test_sample = min(2000, len(os.listdir(os.path.join(data_path, 'test'))))

        # load training set
        x_train = np.empty((self.num_train_samples, self.opt.loadHeight, self.opt.loadWidth, 1), dtype='uint8')
        y_train = np.empty((self.num_train_samples, self.opt.cap_len * self.opt.char_set_len), dtype='uint8')
        train_labels = util.load_label(os.path.join(data_path, label + '_train.txt'))
        for i in range(self.num_train_samples):
            # print(i)
            img_name = os.path.join(data_path, 'train', str(i) + '.jpg')
            x_train[i, :, :, :] = util.load_image(img_name)
            y_train[i, :] = self.text2vec(train_labels[i])

        # load testing set
        x_test = np.empty((self.num_test_sample, self.opt.loadHeight, self.opt.loadWidth, 1), dtype='uint8')
        y_test = np.empty((self.num_test_sample, self.opt.cap_len * self.opt.char_set_len), dtype='uint8')
        test_labels = util.load_label(os.path.join(data_path, label + '_test.txt'))
        for i in range(self.num_test_sample):
            # print(i)
            img_name = os.path.join(data_path, 'test', str(i) + '.jpg')
            x_test[i, :, :, :] = util.load_image(img_name)
            y_test[i, :] = self.text2vec(test_labels[i])

        return (x_train, y_train), (x_test, y_test)

    # Synthetic data generator
    def synth_generator(self, phase):
        if phase == 'train':
            return self.datagen.flow(self.x_train, self.y_train, batch_size=self.opt.batchSize)
        elif phase == 'val':
            return self.datagen.flow(self.x_test, self.y_test, batch_size=self.opt.batchSize, shuffle=False)
        else:
            raise ValueError('Please input train or val phase')

    # Real data generator
    def real_generator(self, phase):
        return self.synth_generator(phase)


# if __name__ == "__main__":

