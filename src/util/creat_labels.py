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


import shutil

# txt_path = '../datasets/google/real/label-check.txt'
# save_path = '../datasets/google/real/label_9.txt'
# save_charNum_path = '../datasets/google/real/char_num.txt'
# img_path = '../datasets/google/real/Google0_4246/'
# img_save_path = '../datasets/google/real/Google_9/'
# i = 0
# j = 509
# with open(txt_path, 'r') as fr:
#     with open(save_path, 'w') as fw:
#         with open(save_charNum_path, 'w') as fwn:
#             lines = ''
#             num_char = ''
#             for line in fr:
#                 if len(line.strip('\n')) == 9:
#                     num_char += str(9) + '#'
#                     lines += line.strip('\n') + '#'
#                     shutil.copyfile(img_path + str(i) + '.jpg', img_save_path + str(j) + '.jpg')
#                     j += 1
#                     print(line)
#                 elif len(line.strip('\n')) == 10:
#                     num_char += str(10) + '#'
#                 else:
#                     num_char += str(8) + '#'
#                 i += 1
#             print(j)
#             fw.write(lines)
#             fwn.write(num_char)


# path = '../datasets/google/real/char_num.txt'
# save_path1 = '../datasets/google/real/char_num_test.txt'
# save_path2 = '../datasets/google/real/char_num_train.txt'
# with open(path, 'r') as fp:
#     with open(save_path1, 'w') as fp1:
#         with open(save_path2, 'w') as fp2:
#             lines = fp.read()
#             characters_num = lines.split('#')
#             chars_train = ''
#             chars_test=''
#             for i in range(4247):
#                 if i < 4000:
#                     chars_train += characters_num[i] + '#'
#                 else:
#                     chars_test += characters_num[i] + '#'
#             fp1.write(chars_test)
#             fp2.write(chars_train)


img_path = '../datasets/google/real/Google0_4246/'
img_save_path = '../datasets/google/real/Google4000-4246/'

j = 0
for i in range(4000,4247,1):
    shutil.copyfile(img_path + str(i) + '.jpg', img_save_path + str(j) + '.jpg')
    j += 1