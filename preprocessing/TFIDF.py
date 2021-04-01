import jieba
import jieba.posseg
from operator import itemgetter

STOP_WORDS = {"the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are", "by", "be", "as", "on",
              "with", "can", "if", "from", "which", "you", "it", "this", "then", "at", "have", "all", "not", "one",
              "has", "or", "that"}


class IDFLoader(object):

    def __init__(self):
        self.path = "./preprocessing/idf.txt"
        self.freq = {}  # 词频
        self.median = 0.0  # 中值
        self.set_path()

    def set_path(self):
        content = open(self.path, 'rb').read().decode('utf-8')
        self.freq = {}
        for line in content.splitlines():
            word, freq = line.strip().split(' ')
            self.freq[word] = float(freq)
        self.median = sorted(
            self.freq.values())[len(self.freq) // 2]

    def get_idf(self):
        return self.freq, self.median


def get_stop_words():
    # print('\n\n\n111\n\n\n')
    with open('./preprocessing/english', 'r', encoding='utf-8') as f:
        # print('\n\n\n111\n\n\n')
        while True:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            # print(s)
            STOP_WORDS.add(s)
    # print(STOP_WORDS)


class TFIDF:

    def __init__(self):
        get_stop_words()
        self.stop_words = STOP_WORDS.copy()
        self.idf_loader = IDFLoader()
        self.freq, self.median = self.idf_loader.get_idf()

    def extract_tags(self, sentence, topK=10, withWeight=False):
        words = jieba.cut(sentence)
        freq = {}
        for w in words:
            wc = w
            if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                continue
            freq[w] = freq.get(w, 0.0) + 1.0
        # 统计词频
        total = sum(freq.values())
        for k in freq:
            kw = k
            freq[k] *= self.freq.get(kw, self.median) / total

        # 是否输出weight
        if withWeight:
            tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(freq, key=freq.__getitem__, reverse=True)

        # 输出几个
        if topK:
            return tags[:topK]
        else:
            return tags

    def extract_tags_from_words(self, words, withWeight=False):
        # words = jieba.cut(sentence)
        freq = {}
        for w in words:
            wc = w
            if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                continue
            freq[w] = freq.get(w, 0.0) + 1.0
        # 统计词频
        total = sum(freq.values())
        for k in freq:
            kw = k
            freq[k] *= self.freq.get(kw, self.median) / total

        # 是否输出weight
        if withWeight:
            tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(freq, key=freq.__getitem__, reverse=True)

        return tags
