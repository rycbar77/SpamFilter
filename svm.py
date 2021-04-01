from sklearn.svm import LinearSVC
from trainingUtils import load_file
import numpy as np
from file_utils import init_test_files
from configs import *
import matplotlib.pyplot as plt


def words2vec(vocabList, inputSet):
    return_vec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            return_vec[vocabList.index(word)] += 1

    return return_vec


# 构造向量空间
def make_vec(_ham_path='./email/ham', _spam_path='./email/spam'):
    cab_ham, _ = load_file(_ham_path)
    cab_spam, _ = load_file(_spam_path)
    _, vocab_ham = load_file(ham_path)
    _, vocab_spam = load_file(spam_path)
    vocab = vocab_ham | vocab_spam
    vo = list(vocab)
    labels = []
    X = []
    for document in cab_ham:
        labels.append(0)
        X.append(words2vec(vo, document))

    for document in cab_spam:
        labels.append(1)
        X.append(words2vec(vo, document))

    X = np.array(X)
    y = np.array(labels)
    return X, y


def test_svm():
    model = LinearSVC()

    init_test_files()
    X, y = make_vec(train_ham_path, train_spam_path)
    X_test, y_test = make_vec(test_ham_path, test_spam_path)

    model.fit(X, y)

    res = model.predict(X_test)
    correct = 0
    for i in range(len(y_test)):
        if res[i] == y_test[i]:
            correct += 1

    print("准确率：%.2f" % (correct / len(res)))
    return correct / len(res)


def svm_unit_test():
    correct = []
    for i in range(50):
        correct.append(test_svm())
    print("平均准确率： %.2f" % (sum(correct) / 50))
    print("\n------------------------------------------\n")
    x = [x for x in range(1, 50 + 1)]
    y = correct[:]
    plt.scatter(x, y)


if __name__ == '__main__':
    svm_unit_test()
    plt.show()
