import os
import random
from parseEmail import text_parse
from parseEmail import mail_parse
from preprocessing.TFIDF import TFIDF
from file_utils import *


# 构造单词集
def create_vocab_list(dataSet):
    vocab_set = set([])
    for document in dataSet:
        vocab_set = vocab_set | set(document)
    return vocab_set


# 读取文件，特征提取
def load_file(path):
    words = []
    tfidf = TFIDF()
    tags = []
    for lists in os.listdir(path):
        fp = os.path.join(path, lists)
        with open(fp, encoding='utf-8', errors='ignore') as f:
            data = f.read()
            # print(data)
            # print(len(textParse(data)))
            tmp = text_parse(data)
            tmp2 = tfidf.extract_tags_from_words(tmp)
            if len(tmp2) > 30:
                tmp2 = tmp2[:30]
            words.append(tmp)
            tags.append(tmp2)
    vocab = create_vocab_list(tags)
    return words, vocab


# 大数据集用
def load_file_from_list(lists):
    words = []
    tags = []
    tfidf = TFIDF()
    for l in lists:
        with open(l, 'rb') as f:
            data = f.read()
            tmp = mail_parse(data)
            tmp2 = tfidf.extract_tags_from_words(tmp)
            if len(tmp2) > 30:
                tmp2 = tmp2[:30]
            words.append(tmp)
            tags.append(tmp2)
    print("loaded %d files!!!" % len(lists))
    vocab = create_vocab_list(tags)
    print("vectorized!!! total: %d" % len(vocab))
    return words, vocab


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
    ham_list = random.sample(ham_list, 1000)
    random.shuffle(ham_list)
    train_size = int(len(ham_list) * 0.8)
    train_ham_list = ham_list[:train_size]
    test_ham_list = ham_list[train_size:]
    spam_list = random.sample(spam_list, 1000)
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
