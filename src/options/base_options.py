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


import argparse
import os
from util import util


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='./datasets/', help='root path to images')
        self.parser.add_argument('--cap_scheme', type=str, default='ebay', help='name of the experiment, it decides where to store samples and models')
        # self.parser.add_argument('--cap_len_category', type=int, default=3, help='number of categories of captcha length')
        self.parser.add_argument('--cap_len', type=int, default=6, help='the length of captcha')
        self.parser.add_argument('--char_set_len', type=int, default=10, help='the numer of char to be used, e.g. weibo: 28, jd: 22, sina: 46, baidu: 25, qihu360: 52, wikipedia and google: 26, ebay: 10, alipay: 31, sohu: 30, live: 34')
        self.parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
        self.parser.add_argument('--train_size', type=int, default=500, help='the number of transferring training set')
        self.parser.add_argument('--loadHeight', type=int, default=55, help='default to this height, e.g. jd: 36, weibo: 50, sina, baidu and qihu360: 40, wikipedia and google: 70, ebay: 55, alipay: 30, sohu: 33, live: 96')
        self.parser.add_argument('--loadWidth', type=int, default=155, help='default to this width, e.g. jd: 150, weibo: 120, sina, baidu, qihu360 and alipay: 100, wikipedia: 250, ebay: 155. sohu: 110, google: 200, live: 216')
        self.parser.add_argument('--keep_prob', type=int, default=0.5, help='default dropout value')
        self.parser.add_argument('--isTune', action='store_true', default=True, help='True: train-error fine tune model; False: train-error base model')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 0,1,2')
        self.parser.add_argument('--model', type=str, default='LeNet5', help='choose which model to use, only exists LeNet5')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are save here')
        self.parser.add_argument('--callbacks', action='store_true', default=True, help='save best val_acc model')
        self.parser.add_argument('--display_winsize', type=int, default=200, help='display window size, this is the width value')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train-error or test

        str_ids = self.opt.gpu_ids.split(',')
        if len(str_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_ids

        args = vars(self.opt)

        print('--------------------Options-------------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('---------------------End----------------------')

        # save to disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.cap_scheme, str(self.opt.train_size))
        util.mkdirs(expr_dir)
        if self.opt.isTrain:
            file_name = os.path.join(expr_dir, 'opt_train.txt')
        else:
            file_name = os.path.join(expr_dir, 'opt_test.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('-----------------Train Options----------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('---------------------End----------------------\n')
        return self.opt
