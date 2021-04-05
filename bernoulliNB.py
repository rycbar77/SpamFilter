from math import log

from multinomialNB import MultinomialNB
import numpy as np
from collections import defaultdict


# 伯努利模型
class BernoulliNB(MultinomialNB):

    def words2vec(self, vocabList, inputSet):
        return_vec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                return_vec[vocabList.index(word)] = 1

        return return_vec

    def update_prior_prob(self, y):
        # P(y=ck)
        class_num = len(self.classes)

        sample_num = len(y)
        for c in self.classes:
            self.prior_prob.append((np.sum(np.equal(y, c)) + self.alpha) / float(sample_num + class_num * self.alpha))

    def update_feature_prob(self, feature, c):
        values = np.unique(feature)
        total = len(feature)
        prob = defaultdict(lambda: (self.alpha / float(total + len(values) * self.alpha)))
        for v in values:
            prob[v] = ((np.sum(np.equal(feature, v)) + self.alpha) / float(total + len(values) * self.alpha))
        return prob

    def predict_sample(self, x):
        label = -1
        max_prob = 0

        # 先验概率*条件概率
        for index in range(len(self.classes)):
            label_tmp = self.classes[index]
            prior = self.prior_prob[index]
            conditional = 1.0
            feature_prob = self.conditional_prob[label_tmp]
            for i in range(len(feature_prob)):
                conditional *= feature_prob[i][x[i]]

            # argmax
            if prior * conditional > max_prob:
                max_prob = prior * conditional
                label = label_tmp

        return label
