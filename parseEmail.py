import _io
import copy
import email
from email.parser import BytesParser, Parser
from email.policy import default
from html import unescape
import re
from nltk.stem.porter import PorterStemmer

STOP_WORDS = {"the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are", "by", "be", "as", "on",
              "with", "can", "if", "from", "which", "you", "it", "this", "then", "at", "have", "all", "not", "one",
              "has", "or", "that"}


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


def text_parse(text):
    import re
    get_stop_words()
    text = re.sub('<[^<>]+>', ' ', text)

    text = re.sub('[0-9]+', ' number ', text)

    # Anything starting with http or https:// replaced with 'httpaddr'
    text = re.sub('(http|https)://[^\s]*', ' http ', text)

    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    text = re.sub('[^\s]+@[^\s]+', ' email ', text)

    # The '$' sign gets replaced with 'dollar'
    text = re.sub('[$]+', ' dollar ', text)

    list_of_tokens = re.split(r'[^\w]', text)
    stemmer = PorterStemmer()

    list_of_tokens = [stemmer.stem(i) for i in list_of_tokens if i != '']

    # print(list_of_tokens)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2 and tok.lower() not in STOP_WORDS]


def html_parse(text):
    import re
    get_stop_words()
    text = re.sub("Content-Type:(.*)\n", '', text)
    text = re.sub("Content-Transfer-Encoding:(.*)\n", '', text)
    text = re.sub('<[^<>]+>', '', text)
    text = re.sub('[0-9]+', ' number ', text)
    list_of_tokens = re.split(r'[^\w]', text)
    stemmer = PorterStemmer()

    list_of_tokens = [stemmer.stem(i) for i in list_of_tokens if i != '']

    # print(list_of_tokens)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2 and tok.lower() not in STOP_WORDS]


def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', ' ', html, flags=re.M | re.S | re.I)
    text = re.sub(r'<a\s.*?>', ' http ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', ' ', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)


def get_file(msg):
    data_char = ''
    for part in msg.walk():
        part_charset = part.get_content_charset()
        # print(part_charset)
        part_type = part.get_content_type()
        # print(part_type)
        if part_type == "text/plain":
            data = part.get_payload(decode=True)
            try:
                data = data.decode(part_charset, errors="replace")
            except:
                data = data.decode('gb2312', errors="replace")
            data = html_to_plain_text(data)
            data_char = data_char + '\n' + data
    return data_char + '\n'


def mail_parse(text):
    # msg = Parser().parsestr(text)
    # type = msg.get_content_type()
    # msg.get_content_charset()
    msg = email.message_from_bytes(text)
    html = get_file(msg)
    # print(html)
    # try:
    #     html = "".join([str(i).replace('_', '') for i in s])
    # except UnicodeEncodeError:
    #     for i in s:
    #         print(i)
    #     return
    # if type != 'text/plain':
    # print(html)
    return html_parse(html)


if __name__ == "__main__":
    with open('./trec07p/data/inmail.897', 'rb') as f:
        s = f.read()
    mail_parse(s)
