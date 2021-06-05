from pyvi import ViTokenizer
import re
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np


def read_file(file_name):
    # return list VietNamese sentences, list English sentences
    with open(file_name + "train.vi", 'r', encoding='utf8') as file:
        vn_list = file.readlines()
    list_vn = [n.replace('\n', '').lower() for n in vn_list]

    with open(file_name + "train.en", 'r', encoding='utf8') as file:
        en_list = file.readlines()
    list_en = [n.replace('\n', '').lower() for n in en_list]
    return list_vn, list_en


def split_dataset(x, y):
    # split dataset
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1,
                                                      random_state=42)
    return x_train, x_val, y_train, y_val


def preprocess_string(s):
    # preprocess string
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)
    return s


def tokenize(x_train, y_train):
    vocab_en = []
    vocab_vn = []
    for sent in x_train:
        for word in sent.split():
            word = preprocess_string(word)
            if word != '':
                vocab_en.append(word)
    for sent in y_train:
        for word in sent.split():
            word = preprocess_string(word)
            if word != '':
                vocab_vn.append(word)
    # return dictionary English
    # index <unk>, <sos>, <pad>, <eos> in vocab English
    # size vocab 10000 word max frequent
    corpus_en = Counter(vocab_en)
    corpus_en = sorted(corpus_en, key=corpus_en.get, reverse=True)[:10000]
    en_dict = {w: i + 4 for i, w in enumerate(corpus_en)}
    en_dict['<unk>'] = 0
    en_dict['<pad>'] = 1
    en_dict['<sos>'] = 2
    en_dict['<eos>'] = 3
    # return dictionary Vietnamese
    # index <unk>, <sos>, <pad>, <eos> in vocab VietNam
    # size vocab 10000 word max frequent
    corpus_vn = Counter(vocab_vn)
    corpus_vn = sorted(corpus_vn, key=corpus_vn.get, reverse=True)[:10000]
    vn_dict = {w: i + 4 for i, w in enumerate(corpus_vn)}
    vn_dict['<unk>'] = 0
    vn_dict['<pad>'] = 1
    vn_dict['<sos>'] = 2
    vn_dict['<eos>'] = 3
    return en_dict, vn_dict


def word_to_index(sentences, vocab):
    list_w_to_index = []
    list_index_to_word = []
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    for sent in sentences:
        list_sent_1 = [2]
        list_sent_2 = ['<sos>']
        for word in sent.split():
            if word not in punc and preprocess_string(word) in vocab.keys():
                list_sent_1.append(vocab[preprocess_string(word)])
                list_sent_2.append(preprocess_string(word))
            if word not in punc and preprocess_string(word) not in vocab.keys():
                list_sent_1.append(vocab['<unk>'])
        list_sent_1.append(3)
        list_sent_2.append('<eos>')
        list_w_to_index.append(list_sent_1)
        list_index_to_word.append(list_sent_2)
    return list_w_to_index, list_index_to_word


def filter_data(x, y, min_len, max_len):
    x_filter = []
    y_filter = []
    for i in range(len(x)):
        if min_len <= len(x[i]) <= max_len:
            x_filter.append(x[i])
            y_filter.append(y[i])
    return x_filter, y_filter


def sort_data_length(x, y):
    pair = zip(x, y)
    a, b = zip(*sorted(pair, key=lambda k: len(k[0])))
    return a, b


def encode_data(x_train, x_val, y_train, y_val, en_dict, vn_dict):
    # return encode sentences to index in English train/valid data
    list_en_train, list_en_train_word = word_to_index(x_train, en_dict)
    list_en_val, _ = word_to_index(x_val, en_dict)
    # return encode sentences to index in VietNamese train/valid data
    list_vn_train, list_vn_train_word = word_to_index(y_train, vn_dict)
    list_vn_val, _ = word_to_index(y_val, vn_dict)
    # filter data
    list_en_train, list_vn_train = filter_data(list_en_train, list_vn_train, min_len=3, max_len=1000)
    list_en_val, list_vn_val = filter_data(list_en_val, list_vn_val, min_len=3, max_len=1000)
    # sort pair data length
    list_en_train, list_vn_train = sort_data_length(list_en_train, list_vn_train)
    list_en_val, list_vn_val = sort_data_length(list_en_val, list_vn_val)

    return list_en_train, list_en_val, list_vn_train, list_vn_val


def padding_src(sentences_src, seq_len):
    features_src = np.ones((len(sentences_src), seq_len), dtype=int)
    for ii, sentences in enumerate(sentences_src):
        features_src[ii, 0:len(sentences)] = np.array(sentences)[:seq_len]
        features_src[ii, len(sentences) - 1] = 1
        features_src[ii, -1] = 3
    return features_src


def padding_tag(sentences_tag, seq_len):
    # padding for y_train list sequences Viet_Nam
    features_tag = np.ones((len(sentences_tag), seq_len), dtype=int)
    for ii, sentences in enumerate(sentences_tag):
        features_tag[ii, 0:len(sentences)] = np.array(sentences)[:seq_len]
    return features_tag


if __name__ == '__main__':
    '''
    file_name = "/Users/nguyenquan/Desktop/mars_project/neuron_network/NLP_model/data/"
    list_vn, list_en = read_file(file_name)
    x_train, x_val, y_train, y_val = split_dataset(list_en, list_vn)
    print(len(x_train))
    print((len(x_val)))
    # print(x_train[0:2])
    # print(y_train[0:2])

    en_dict, vn_dict = tokenize(x_train, y_train)
    print(en_dict)
    print(vn_dict)
    list_en_train, list_en_val, list_vn_train, list_vn_val \
        = encode_data(x_train, x_val, y_train, y_val, en_dict, vn_dict)
    list_en_train, list_vn_train = sort_data_length(list_en_train, list_vn_train)
    list_en_val, list_vn_val = sort_data_length(list_en_val, list_vn_val)
    print(list_en_train[0:5])
    print(list_vn_train[0:5])
    print(list_en_val[0:5])
    print(list_vn_val[0:5])
    '''
    a = [[1, 2, 3, 3], [4, 6, 7, 8, 9]]
    print(padding_tag(a, 10))
