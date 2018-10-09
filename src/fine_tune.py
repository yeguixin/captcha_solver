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


from options.tune_options import TuneOptions
from generator.generator import Generator
from models.models import creat_model
from keras.optimizers import adam

# NB_RETRAIN_LAYERS = 20
# epochs = 200

opt = TuneOptions().parse()
# opt.isTune = True
# opt.batchSize = 32
generator = Generator(opt)
train_generator = generator.real_generator('train')
val_generator = generator.real_generator('val')

num_train_samples = generator.num_train_samples
num_test_sample = generator.num_test_sample

model = creat_model(opt)
model.load_weight()

# setup_to_finetune(model)
# model.setup_to_finetune()

# history_ft = model.fit_generator(
#     train_generator,
#     steps_per_epoch=num_train_samples // opt.batchSize,
#     epochs=opt.epoch,
#     validation_data=val_generator,
#     validation_steps=num_test_sample // opt.batchSize
# )
#
# model.save(history_ft)

test_data = [generator.x_test, generator.y_test]
# test_data = [generator.x_train, generator.y_train]
model.predict(test_data, batch_size=opt.batchSize)

# if opt.plot:
#     model.plot_training(history_ft)
