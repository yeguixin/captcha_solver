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


def creat_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'LeNet5':
        from .lenet import LeNetModel
        model = LeNetModel()
    else:
        raise ValueError('Model [%s] not recognized.' % opt.model)
    model.initialize(opt)
    return model
