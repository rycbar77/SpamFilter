from multinomialNB import MultinomialNB


# 伯努利模型
class BernoulliNB(MultinomialNB):

    def words2vec(self, vocabList, inputSet):
        return_vec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                return_vec[vocabList.index(word)] = 1

        return return_vec
