import os
from parseEmail import text_parse
from parseEmail import mail_parse
from preprocessing.TFIDF import TFIDF


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
