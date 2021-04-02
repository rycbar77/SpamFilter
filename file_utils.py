import os
import random
import shutil
import chardet
from configs import *


# 随机选择数据集，划分训练集和测试集
def init_bigdata_test_files():
    # file_list = get_files(bigdata_path)
    ham_list = []
    spam_list = []
    with open(bigdata_index_path, 'r') as f:
        while True:
            s = f.readline()
            if not s:
                break
            label, file = s.split(' ')
            file = file.replace('../data', bigdata_path).strip()
            if label == 'ham':
                ham_list.append(file)
            else:
                spam_list.append(file)
    ham_list = random.sample(ham_list, 500)
    random.shuffle(ham_list)
    train_size = int(len(ham_list) * 0.8)
    train_ham_list = ham_list[:train_size]
    test_ham_list = ham_list[train_size:]
    spam_list = random.sample(spam_list, 500)
    random.shuffle(spam_list)
    train_size = int(len(spam_list) * 0.8)
    train_spam_list = spam_list[:train_size]
    test_spam_list = spam_list[train_size:]

    return train_ham_list, train_spam_list, test_ham_list, test_spam_list


def init_test_files():
    convert_dir(ham_path)
    convert_dir(spam_path)
    ham_list = get_files(ham_path)
    spam_list = get_files(spam_path)

    random.shuffle(ham_list)
    train_size = int(len(ham_list) * 0.8)
    train_ham_list = ham_list[:train_size]
    test_ham_list = ham_list[train_size:]

    random.shuffle(spam_list)
    train_size = int(len(spam_list) * 0.8)
    train_spam_list = spam_list[:train_size]
    test_spam_list = spam_list[train_size:]

    # print(train_spam_list)
    recreate_dir(train_path)
    recreate_dir(test_path)

    os.mkdir(train_ham_path)
    os.mkdir(train_spam_path)
    os.mkdir(test_ham_path)
    os.mkdir(test_spam_path)

    copy_files(test_ham_path, test_ham_list, ham_path)
    copy_files(test_spam_path, test_spam_list, spam_path)
    copy_files(train_spam_path, train_spam_list, spam_path)
    copy_files(train_ham_path, train_ham_list, ham_path)

    return train_ham_list, train_spam_list, test_ham_list, test_spam_list


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


if __name__ == "__main__":
    init_bigdata_test_files()
