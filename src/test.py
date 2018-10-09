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


from options.test_options import TestOptions
import os
from models.models import creat_model
# from keras.models import load_model
from generator.generator import Generator


opt = TestOptions().parse()
# model_path = os.path.join(opt.checkpoints_dr, opt.cap_scheme, opt.model_name)
model = creat_model(opt)
model.load_weight()

generator = Generator(opt)
# test_generator = generator.synth_generator(opt.phase)
test_data = (generator.x_test, generator.y_test)
# input_image = test_data[0][20:21, :, :, :]
# print(input_image.shape)
# test_data = (generator.x_train, generator.y_train)
# model.visualize_model(input_image, 2)

# num_test_samples = generator.num_test_sample
# steps = num_test_samples // opt.batchSize

# model.predict(test_data, batch_size=len(test_data[0]))
model.predict(test_data, batch_size=opt.batchSize)
# model.predict_generator(test_generator, steps=steps)
# acc, predicts, real =
