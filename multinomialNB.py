from collections import defaultdict
from math import log

from trainingUtils import *

import numpy as np


# 多项式模型
class MultinomialNB(object):

    def __init__(self, alpha=1.0):
        self.word_count = 0
        self.vocab = []
        self.prior_prob = []
        self.alpha = alpha
        self.classes = None
        self.conditional_prob = None
        self.count = []

    def words2vec(self, vocabList, inputSet):
        return_vec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                return_vec[vocabList.index(word)] += 1

        return return_vec

    # 构造向量空间
    def make_vec(self, ham_path='./email/ham', spam_path='./email/spam', train=False):
        words_ham, vocab_ham = load_file(ham_path)
        words_spam, vocab_spam = load_file(spam_path)
        self.count.append(len(words_ham))
        self.count.append(len(words_spam))
        self.word_count = self.count[0] + self.count[1]
        return self.vectorize(words_ham, words_spam, vocab_ham, vocab_spam, train)

    def make_vec_from_list(self, ham_list, spam_list, train=False):
        print("loading ham files...")
        words_ham, vocab_ham = load_file_from_list(ham_list)
        print("loading spam files...")
        words_spam, vocab_spam = load_file_from_list(spam_list)
        self.count.append(len(words_ham))
        self.count.append(len(words_spam))
        self.word_count = self.count[0] + self.count[1]
        return self.vectorize(words_ham, words_spam, vocab_ham, vocab_spam, train)

    def vectorize(self, words_ham, words_spam, vocab_ham, vocab_spam, train):
        tmp = set([])
        if train:
            self.vocab = vocab_ham | vocab_spam
        else:
            tmp = (vocab_ham | vocab_spam) - self.vocab
        vo = list(self.vocab) + list(tmp)
        labels = []
        X = []
        for document in words_ham:
            labels.append(0)
            X.append(self.words2vec(vo, document))

        for document in words_spam:
            labels.append(1)
            X.append(self.words2vec(vo, document))

        X = np.array(X)
        y = np.array(labels)
        return X, y

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.update_prior_prob(y)
        # P( xj | y=ck )
        self.conditional_prob = {}
        for c in self.classes:
            self.conditional_prob[c] = defaultdict(lambda: defaultdict(lambda: log(0.5)))
            for i in range(len(X[0])):
                feature = X[np.equal(y, c)][:, i]
                # print(np.equal(y,c))
                # print(X[np.equal(y, c)])
                self.conditional_prob[c][i] = self.update_feature_prob(feature, c)
        return self

    def update_prior_prob(self, y):
        # P(y=ck)
        class_num = len(self.classes)

        sample_num = self.word_count
        for c in self.classes:
            self.prior_prob.append(log(self.count[c] + self.alpha) - log(sample_num + class_num * self.alpha))

    def update_feature_prob(self, feature, c):
        values = np.unique(feature)
        total = self.count[c]
        prob = defaultdict(lambda: log(self.alpha) - log(total + len(values) * self.alpha))
        for v in values:
            prob[v] = log(np.sum(np.equal(feature, v)) + self.alpha) - log(total + len(values) * self.alpha)
            # print(prob[v])
        return prob

    def predict_sample(self, x):
        label = -1
        max_prob = -999

        # 先验概率*条件概率
        for index in range(len(self.classes)):
            label_tmp = self.classes[index]
            prior = self.prior_prob[index]
            conditional = 0.0
            feature_prob = self.conditional_prob[label_tmp]
            # print(feature_prob)
            for i in range(len(feature_prob)):
                conditional += feature_prob[i][x[i]]
                # if feature_prob[i][x[i]] > 0:
                #     print(feature_prob[i][x[i]])

            # argmax
            if prior + conditional > max_prob:
                max_prob = prior + conditional
                label = label_tmp

        return label

    def predict(self, X):
        labels = []
        for i in range(X.shape[0]):
            label = self.predict_sample(X[i])
            labels.append(label)
        return labels
