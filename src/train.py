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


from options.train_options import TrainOptions
from generator.generator import Generator
from models.models import creat_model

opt = TrainOptions().parse()
generator = Generator(opt)
train_generator = generator.synth_generator('train')
val_generator = generator.synth_generator('val')
# img_batch = val_generator.__next__()
model = creat_model(opt)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=generator.num_train_samples // opt.batchSize,
    epochs=opt.epoch,
    validation_data=val_generator,
    validation_steps=generator.num_test_sample // opt.batchSize,
    class_weight='auto'
)
# save model
model.save(history)

# plot validation accuracy and loss
if opt.plot:
    model.plot_training(history)
