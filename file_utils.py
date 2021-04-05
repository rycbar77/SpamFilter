import os
import random
import shutil
import chardet
from configs import *


# 遍历文件
def get_files(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(file)
    return file_list


# 重新创建文件夹
def recreate_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)


# 复制文件
def copy_files(dest_dir, path_list, src_dir):
    for path in path_list:
        src_path = src_dir + path
        dest_path = dest_dir + path
        shutil.copyfile(src_path, dest_path)


# 修改编码
def convert_format(file):
    with open(file, 'rb+')as f:
        content = f.read()
        encode = chardet.detect(content)['encoding']
        if encode != 'utf-8':
            try:
                gbk_content = content.decode(encode)
                utf_byte = bytes(gbk_content, encoding='utf-8')
                f.seek(0)
                f.write(utf_byte)
            except IOError:
                print('fail')


def convert_dir(path):
    for file in os.listdir(path):
        file = os.path.join(path, file)
        convert_format(file)

#
# if __name__ == "__main__":
#     init_bigdata_test_files()
