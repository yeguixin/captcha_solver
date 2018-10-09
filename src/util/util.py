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


import os
from PIL import Image
import numpy as np


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_image(path):
    with Image.open(path).convert('L') as img:
        img = np.array(img)
        img = np.expand_dims(img, axis=2)
        return img


def load_label(txt):
    with open(txt, 'r') as fp_txt:
        labels = fp_txt.read()
        labels = labels.split('#')
        return labels


def char2dict(char_set):
    dict = {}
    for i, char in enumerate(char_set):
        dict[i] = char
    return dict


def create_dict(cap_scheme):
    char_set = []
    scheme = CaptchaSchemes()
    if cap_scheme == 'weibo':
        char_set = scheme.weibo
    elif cap_scheme == 'jd':
        char_set = scheme.jd
    elif cap_scheme == 'sina':
        char_set = scheme.sina
    elif cap_scheme == 'baidu':
        char_set = scheme.baidu
    elif cap_scheme == 'qihu':
        char_set = scheme.qihu
    elif cap_scheme == 'wiki':
        char_set = scheme.wiki
    elif cap_scheme == 'ebay':
        char_set = scheme.ebay
    elif cap_scheme == 'alipay':
        char_set = scheme.alipay
    elif cap_scheme == 'sohu':
        char_set = scheme.sohu
    elif cap_scheme == 'google':
        char_set = scheme.google
    elif cap_scheme == 'live':
        char_set = scheme.live
    return char2dict(char_set)


def vect2text(vect, dict):
    text = ''
    for i, key in enumerate(vect):
        value = dict[key]
        text += value
    return text


def print_predict(opt):
    scheme = CaptchaSchemes()
    preds = opt.preds
    # real_labels = opt.reals
    char_set = []
    total_num = len(preds)
    print('total_num: {}'.format(total_num))
    preds_right_num = 0
    if opt.cap_scheme == 'weibo':
        char_set = scheme.weibo
    elif opt.cap_scheme == 'jd':
        char_set = scheme.jd
    elif opt.cap_scheme == 'sina':
        char_set = scheme.sina
    elif opt.cap_scheme == 'baidu':
        char_set = scheme.baidu
    elif opt.cap_scheme == 'qihu':
        char_set = scheme.qihu
    elif opt.cap_scheme == 'wiki':
        char_set = scheme.wiki
    elif opt.cap_scheme == 'ebay':
        char_set = scheme.ebay
    elif opt.cap_scheme == 'alipay':
        char_set = scheme.alipay
    elif opt.cap_scheme == 'sohu':
        char_set = scheme.sohu
    elif opt.cap_scheme == 'google':
        char_set = scheme.google
    elif opt.cap_scheme == 'live':
        char_set = scheme.live
    dict = char2dict(char_set)
    if opt.isTune:
        label = 'real'
    else:
        label = 'synthetic'
    if opt.phase == 'val':
        real_label_name = os.path.join(opt.dataroot, opt.cap_scheme, label, label + '_test.txt')
    elif opt.phase == 'train':
        real_label_name = os.path.join(opt.dataroot, opt.cap_scheme, label, label + '_train.txt')
    else:
        real_label_name = os.path.join(opt.dataroot, opt.cap_scheme, label, label + '_test.txt')
    real_labels = load_label(real_label_name)
    for i, pred in enumerate(preds):
        pred = vect2text(pred, dict)
        if pred.lower() == real_labels[i].lower():
            preds_right_num += 1
            print('------Correct Prediction-------')
        print('No.{} \t Predict: {} \t Real: {}' .format(i, pred, real_labels[i]))
        # print('Predict: {} \t Real: {}'.format(pred, real_labels[i]))
    print('Recognization Accytacy: {}' .format(preds_right_num/total_num))


class CaptchaSchemes:
    def __init__(self):
        self.weibo = [
            '2', '3', '4', '6', '7', '8', '9',
            'A', 'B', 'C', 'E', 'F', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T',
            'V', 'W', 'X', 'Y', 'Z'
        ]
        self.jd = [
            '3', '4', '5', '6', '8',
            'A', 'B', 'C', 'E', 'F', 'H', 'K', 'M', 'N', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y'
        ]
        self.sina = [
            '2', '3', '4', '5', '6', '7', '8',
            'a', 'b', 'c', 'd', 'e', 'f', 'h', 'k', 'm', 'n', 'p', 'q', 's', 'u',
            'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'E', 'F', 'G', 'H', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'U',
            'V', 'W', 'X', 'Y', 'Z'
        ]
        self.baidu = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
        ]
        self.qihu = [
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p',
            'r', 's', 'u', 'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N',
            'P', 'Q', 'R', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        self.wiki = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
        self.ebay = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]
        self.alipay = [
            '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        self.sohu = [
            '2', '3', '4', '5', '6', '7', '8', '9',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y'
        ]
        self.google = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
        self.live = [
            '3', '4', '5', '6',
            'd', 'p', 's', 'y',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]


if __name__ == '__main__':
    dict = create_dict('weibo')
    c = 'Z'
    def char2pos(c):
        k = -1
        for (key, value) in dict.items():
            if value == c:
                k = key
        return k
    key = char2pos(c)
    print(key)
