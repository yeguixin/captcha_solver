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


from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results', help='save results here')
        self.parser.add_argument('--phase', type=str, default='val', help='the dataset folder, e.g. train, test, val')
        self.parser.add_argument('--base_model_name', type=str, default='ebay-improvement-1238-0.83.hdf5', help='which model to load')
        self.parser.add_argument('--how_many', type=int, default=200, help='how many test images to run')
        self.isTrain = False
