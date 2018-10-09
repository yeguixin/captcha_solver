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


from keras import models, layers, optimizers, losses
import os
import sys
import numpy as np
from util import util
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras import backend as K
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LeNetModel:
    def name(self):
        return 'LeNet5 Model'

    def initialize(self, opt):
        self.opt = opt
        self.batchSize = opt.batchSize
        # self.input_image_tensor = layers.Input(shape=(opt.loadHeight, opt.loadWidth, 1))
        self.keep_prob = opt.keep_prob
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.cap_scheme, str(opt.train_size))
        # Load/Define Model
        self.model = self.define_lenet5()

        if self.opt.isTrain:
            # Set optimizer
            # sgd = optimizers.RMSprop(lr=self.opt.lr, decay=1e-6)
            adam = optimizers.adam(lr=opt.lr)
            # Compile model
            # self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
        if self.opt.callbacks:
            bast_model_name = os.path.join(self.save_dir,
                                           self.opt.cap_scheme + '-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
            checkpoint = ModelCheckpoint(bast_model_name,
                                         monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            self.callbacks_list = [checkpoint]
        else:
            self.callbacks_list = None


    # def categorical_crossentropy(self, y_true, y_pred):
    #     # idxs = np.argmax(y_pred.reshape(-1, self.opt.cap_len, self.opt.char_set_len), axis=2)
    #     print(y_pred[0:])
    #     print(type(y_pred))
    #     print(len(y_pred))
    #     idx = np.argpartition(y_pred, -self.opt.cap_len)[-self.opt.cap_len:]
    #     pred = np.zeros(self.opt.cap_len * self.opt.char_set_len)
    #     pred[idx] = 1
    #     return losses.categorical_crossentropy(y_true, pred)

    def define_lenet5(self):
        model = models.Sequential()
        # 5 Convolutional Layers
        model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                                input_shape=(self.opt.loadHeight, self.opt.loadWidth, 1)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(layers.Dropout(self.keep_prob))

        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(layers.Dropout(self.keep_prob))

        model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(layers.Dropout(self.keep_prob))

        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(layers.Dropout(self.keep_prob))

        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(layers.Dropout(self.keep_prob))

        # Fully Connected Layers
        model.add(layers.Flatten())
        model.add(layers.Dense(3072))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(self.keep_prob))

        # model.add(layers.Dense(2048))
        # model.add(layers.Activation('relu'))
        # model.add(layers.Dropout(self.keep_prob))

        model.add(layers.Dense(self.opt.cap_len * self.opt.char_set_len))
        model.add(layers.Activation('softmax'))
        # model.summary()
        return model

    # @ overwrite model.fit_generator
    def fit_generator(self,
                      generator,
                      steps_per_epoch=20,
                      epochs=20,
                      validation_data=None,
                      validation_steps=None,
                      class_weight='auto',
                      callbacks = None
                      ):
        return self.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            class_weight=class_weight,
            callbacks=callbacks
        )

    def predict(self, test_data, batch_size=32):
        # preds = self.model.predict_generator(
        #     generator=generator,
        #     steps=steps,
        #     max_queue_size=max_queue_size
        # )
        x_test, y_test = test_data[0], test_data[1]
        # Normalize data
        x_test = (x_test - 255 / 2) / (255 / 2)
        preds = self.model.predict(x=x_test, batch_size=batch_size)
        # print(preds[9])
        # neuro_no = range(22 * 4)
        # plt.figure()
        # plt.plot(neuro_no, preds[9], 'b-', label='train_loss')
        # plt.show()
        preds = np.argmax(preds.reshape(-1, self.opt.cap_len, self.opt.char_set_len), axis=2)
        self.opt.preds = preds
        # print(self.opt.preds)
        self.opt.reals = y_test
        util.print_predict(self.opt)
        # acc, pred_label, real_label =

    def predict_generator(self, generator, steps=1, max_queue_size=100):
        preds = self.model.predict_generator(
            generator=generator,
            steps=steps,
            max_queue_size=max_queue_size
        )
        preds = np.argmax(preds.reshape(-1, self.opt.cap_len, self.opt.char_set_len), axis=2)
        self.opt.preds = preds
        # self.opt.reals = y_test
        util.print_predict(self.opt)

    def save_model(self):
        if self.opt.isTune:
            model_checkpoint_base_name = os.path.join(self.save_dir, self.opt.cap_scheme + '_finetune.model')
        else:
            model_checkpoint_base_name = os.path.join(self.save_dir, self.opt.cap_scheme + '_org.model')
        # util.mkdirs(model_checkpoint_base_name)
        plot_model(self.model, to_file=model_checkpoint_base_name + '.png', show_shapes=True)
        self.model.save(model_checkpoint_base_name)


    def load_weight(self):
        model_checkpoint_base_name = os.path.join(self.save_dir, self.opt.base_model_name)
        # model_checkpoint_base_name = os.path.join(self.opt.checkpoints_dir, self.opt.cap_scheme, self.opt.base_model_name)
        # print(model_checkpoint_base_name)
        if self.opt.isTune:
            self.model.load_weights(model_checkpoint_base_name)
        else:
            self.model.load_weights(model_checkpoint_base_name)

    def save(self, history):
        # save loss to opt_train.txt
        print('Print the training history:')
        # print(history.history)
        with open(self.save_dir + '/opt_train.txt', 'a') as opt_file:
            opt_file.write(str(history.history))

        self.save_model()

    def setup_to_finetune(self):
        print('layers number of the model {} ' .format(len(self.model.layers)))
        for layer in self.model.layers[:self.opt.nb_retain_layers]:
            layer.trainable = False
        for layer in self.model.layers[self.opt.nb_retain_layers:]:
            weights = layer.get_weights()
            weights = weights * 0
            # # layer.set_weights(weights)
            layer.trainable = True
            # weights = layer.get_weights()
        self.model.compile(optimizer=optimizers.adam(lr=self.opt.lr),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    def visualize_model(self, input_images, layer_index):
        output_layer = self.model.layers[layer_index].output
        input_layer = self.model.layers[0].input
        output_fn = K.function([input_layer], [output_layer])
        output_image = output_fn([input_images])[0]
        print("Output image shape:", output_image.shape)

        fig = plt.figure()
        plt.title("%dth convolutional later view of output" % layer_index)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

        for i in range(32):
            ax = fig.add_subplot(4, 8, i+1)
            im = ax.imshow(output_image[0, :, :, i], cmap='Greys')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([1, 0.07, 0.05, 0.821])
        fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()

        plt.show()

    def plot_training(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        print("run to here")
        print(matplotlib.get_backend())

        plt.plot(epochs, acc, 'r.', label='train_acc')
        plt.plot(epochs, val_acc, 'b', label='val_acc')
        plt.title("Training and validation accuracy")
        plt.legend(loc=0, ncol=2)
        plt.savefig('./accuracy.png')

        plt.figure()
        plt.plot(epochs, loss, 'r.', label='train_loss')
        plt.plot(epochs, val_loss, 'b-', label='val_loss')
        plt.title("Training and validation loss")
        plt.legend(loc=0, ncol=2)
        plt.savefig('./loss.png')
        plt.show()
