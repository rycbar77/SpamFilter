from collections import defaultdict
from trainingUtils import *

import numpy as np


# 多项式模型
class MultinomialNB(object):

    def __init__(self, alpha=1.0):
        self.vocab = []
        self.class_prior = []
        self.alpha = alpha
        self.classes = None
        self.conditional_proba = None

    def feature_proba(self, feature):
        values = np.unique(feature)
        total_num = float(len(feature))
        value_prob = defaultdict(lambda: self.alpha / (total_num + len(values) * self.alpha))
        for v in values:
            value_prob[v] = ((np.sum(np.equal(feature, v)) + self.alpha) / (total_num + len(values) * self.alpha))
        return value_prob

    def fit(self, X, y):
        self.classes = np.unique(y)
        # P(y=ck)

        class_num = len(self.classes)

        sample_num = float(len(y))
        for c in self.classes:
            c_num = np.sum(np.equal(y, c))
            self.class_prior.append((c_num + self.alpha) / (sample_num + class_num * self.alpha))

        # P( xj | y=ck )
        self.conditional_proba = {}
        for c in self.classes:
            self.conditional_proba[c] = {}
            for i in range(len(X[0])):
                feature = X[np.equal(y, c)][:, i]
                # print(np.equal(y,c))
                # print(X[np.equal(y, c)])
                self.conditional_proba[c][i] = self.feature_proba(feature)
        return self

    def get_xj_proba(self, values_proba, target):
        return values_proba[target]

    def predict_single(self, x):
        label = -1
        max_prob = 0

        # 先验概率*条件概率
        for c_index in range(len(self.classes)):
            current_class_prior = self.class_prior[c_index]
            conditional_proba = 1.0
            feature_prob = self.conditional_proba[self.classes[c_index]]
            j = 0
            for feature_i in feature_prob.keys():
                conditional_proba *= self.get_xj_proba(feature_prob[feature_i], x[j])
                j += 1

            if current_class_prior * conditional_proba > max_prob:
                max_prob = current_class_prior * conditional_proba
                label = self.classes[c_index]

        return label

    def predict(self, X):
        labels = []
        for i in range(X.shape[0]):
            label = self.predict_single(X[i])
            labels.append(label)
        return labels

    def words2vec(self, vocabList, inputSet):
        return_vec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                return_vec[vocabList.index(word)] += 1

        return return_vec

    # 构造向量空间
    def make_vec(self, ham_path='./email/ham', spam_path='./email/spam', train=False):
        cab_ham, vocab_ham = load_file(ham_path)
        cab_spam, vocab_spam = load_file(spam_path)
        return self.vectorize(cab_ham, cab_spam, vocab_ham, vocab_spam, train)

    def make_vec_from_list(self, ham_list, spam_list, train=False):
        cab_ham, vocab_ham = load_file_from_list(ham_list)
        cab_spam, vocab_spam = load_file_from_list(spam_list)
        return self.vectorize(cab_ham, cab_spam, vocab_ham, vocab_spam, train)

    def vectorize(self, cab_ham, cab_spam, vocab_ham, vocab_spam, train):
        tmp = set([])
        if train:
            self.vocab = vocab_ham | vocab_spam
        else:
            tmp = (vocab_ham | vocab_spam) - self.vocab
        vo = list(self.vocab) + list(tmp)
        labels = []
        X = []
        for document in cab_ham:
            labels.append(0)
            X.append(self.words2vec(vo, document))

        for document in cab_spam:
            labels.append(1)
            X.append(self.words2vec(vo, document))

        X = np.array(X)
        y = np.array(labels)
        return X, y
