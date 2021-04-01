from multinomialNB import MultinomialNB
from bernoulliNB import BernoulliNB
from gaussianNB import GaussianNB
from file_utils import init_test_files
from file_utils import get_files
from file_utils import init_bigdata_test_files
from configs import *
import matplotlib.pyplot as plt
from svm import svm_unit_test


# 单个测试
def test(model='mul'):
    if model == 'bern':
        test_model = BernoulliNB(alpha=1.0)
    elif model == 'gau':
        test_model = GaussianNB(alpha=1.0)
    else:
        test_model = MultinomialNB(alpha=1.0)
    init_test_files()
    X, y = test_model.make_vec(train_ham_path, train_spam_path, True)
    test_model.fit(X, y)
    X_test, y_test = test_model.make_vec(test_ham_path, test_spam_path)
    res = test_model.predict(X_test)
    correct = 0
    log.write("***\n")
    for i in range(len(y_test)):
        if res[i] == y_test[i]:
            correct += 1
        else:
            ham_list = ['/ham/' + i for i in get_files(test_ham_path)]
            spam_list = ['/spam/' + i for i in get_files(test_spam_path)]
            log.write("failed: %s\n" % (spam_list[i - 5] if i >= 5 else ham_list[i]))
    print("准确率：%.2f" % (correct / len(res)))
    return correct / len(res)


# 50个为一组进行测试
def unit_test(model='mul'):
    if model == 'bern':
        print("\n---------------BernoulliNB----------------\n")
    elif model == 'gau':
        print("\n---------------GaussianNB-----------------\n")
    else:
        print("\n--------------MultinomialNB---------------\n")
    correct = []
    for i in range(50):
        correct.append(test(model))
    print("平均准确率： %.2f" % (sum(correct) / 50))
    print("\n------------------------------------------\n")
    x = [x for x in range(1, 50 + 1)]
    y = correct[:]
    plt.scatter(x, y)


# 基于大数据集进行测试
def test_bigdata(model='mul'):
    if model == 'bern':
        test_model = BernoulliNB(alpha=1.0)
    elif model == 'gau':
        test_model = GaussianNB(alpha=1.0)
    else:
        test_model = MultinomialNB(alpha=1.0)
    print("Random pick!!!")
    train_ham_list, train_spam_list, test_ham_list, test_spam_list = init_bigdata_test_files()
    print("Time to load train data!")
    X, y = test_model.make_vec_from_list(train_ham_list, train_spam_list, True)
    print("Time to fit!!!")
    test_model.fit(X, y)
    print("Fit done!!!\nTime to load test data!")
    X_test, y_test = test_model.make_vec_from_list(test_ham_list, test_spam_list)
    print("Predicting!!!")
    res = test_model.predict(X_test)
    correct = 0
    log.write("***\n")
    print("CheckCheck!!!")
    for i in range(len(y_test)):
        if res[i] == y_test[i]:
            correct += 1
        # else:
        #     log.write("failed: %s\n" % (test_spam_list[i - 5044] if i >= 5044 else test_ham_list[i]))
    print("准确率：%.2f" % (correct / len(res)))
    return correct / len(res)


if __name__ == '__main__':
    log = open('log.txt', 'w+')
    log.write("---------------------------------------------\n\n")
    # unit_test('gau')
    unit_test('mul')
    unit_test('bern')
    svm_unit_test()
    plt.show()
    # test_bigdata('bern')
    # test_bigdata()
    log.write("---------------------------------------------\n\n")
    log.close()
    # bigdata_test_mul()
