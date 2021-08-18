"""
load dataset
"""
import re
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import pickle

path = "/Users/nguyenquan/Desktop/mars_project/neural_machine_translation/data/train_en2vi/"


def read_file(file_name):
    # return list VietNamese sentences, list English sentences
    # đưa ra chuỗi tiếng anh và bản dịch tiếng việt tương ứng
    with open(file_name + "train.vi", 'r', encoding='utf8') as file:
        vn_list = file.readlines()
    list_vn = [n.replace('\n', '').lower() for n in vn_list]

    with open(file_name + "train.en", 'r', encoding='utf8') as file:
        en_list = file.readlines()
    list_en = [n.replace('\n', '').lower() for n in en_list]
    return list_en, list_vn


def split_dataset(x, y):
    # split dataset
    # Chia dữ liệu thành hai thành phần: dữ liệu huấn luyện và dữ liệu đánh giá
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1,
                                                      random_state=42)
    return x_train, x_val, y_train, y_val


def preprocess_string(s):
    # preprocess string
    # hàm xử lý chuỗi string s
    # Remove all non-word characters (everything except numbers and letters)
    # xoá các kí tự không phải là từ, ngoại trừ số và chữ cái
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    # thay thế ccas khoảng trắng bằng không khoảng
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    # xoá các chữ số thành các khoảng trắng
    s = re.sub(r"\d", '', s)
    return s


def tokenize(en_train, vi_train):
    # return dictionary english and dictionary vietnamese
    # Đưa ra tập từ điển Tiếng Anh và tập từ điển Tiếng Việt từ bộ dữ
    vocab_en = []
    vocab_vn = []
    for sent in en_train:
        for word in sent.split():
            word = preprocess_string(word)
            if word != '':
                vocab_en.append(word)
    for sent in vi_train:
        for word in sent.split():
            word = preprocess_string(word)
            if word != '':
                vocab_vn.append(word)
    # return 10000 word max frequent
    # Đưa ra 10000 từ phổ biến nhất xuất hiện trong dữ liệu
    corpus_en = Counter(vocab_en)
    corpus_en = sorted(corpus_en, key=corpus_en.get, reverse=True)[:10000]
    corpus_vn = Counter(vocab_vn)
    corpus_vn = sorted(corpus_vn, key=corpus_vn.get, reverse=True)[:10000]

    return corpus_en, corpus_vn


def dict_laguage(en_train, vn_train):
    # add index <unk>, <sos>, <pad>, <eos> into vocab english and vietnamese
    # Thêm các kí tự đặc biệt vào tập từ điển tiếng Anh và tiếng Việt
    corpus_en, corpus_vn = tokenize(en_train, vn_train)
    en_dict = {w: i + 4 for i, w in enumerate(corpus_en)}
    en_dict['<unk>'], en_dict['<pad>'], en_dict['<sos>'], en_dict['<eos>'] = 0, 1, 2, 3
    vn_dict = {w: i + 4 for i, w in enumerate(corpus_vn)}
    vn_dict['<unk>'], vn_dict['<pad>'], vn_dict['<sos>'], vn_dict['<eos>'] = 0, 1, 2, 3
    return en_dict, vn_dict


def word_to_index(sentences, vocab):
    # return word in sentences to index and index to word
    # Chuyển đổi câu thành index và ngược lại chuyển đổi index thành một câu
    list_w_to_index = []
    list_index_to_w = []
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    for sent in sentences:
        list_sent_1 = [2]
        list_sent_2 = ['<sos>']
        for word in sent.split():
            # Nếu các từ không phải là dấu câu và thuộc trong tập từ điển
            if word not in punc and preprocess_string(word) in vocab.keys():
                list_sent_1.append(vocab[preprocess_string(word)])
                list_sent_2.append(preprocess_string(word))
            # Nếu các từ không phải là dấu câu và không thuộc trong tập từ điển
            if word not in punc and preprocess_string(word) not in vocab.keys():
                list_sent_1.append(vocab['<unk>'])
                list_sent_2.append(preprocess_string(word))
        list_sent_1.append(3)
        list_sent_2.append('<eos>')
        list_w_to_index.append(list_sent_1)
        list_index_to_w.append(list_sent_2)
    return list_w_to_index, list_index_to_w


def encode_data(x_train, x_val, y_train, y_val, en_dict, vn_dict):
    # return encode sentences to index in English train/valid data
    # mã hoá dữ liệu các câu Tiếng Anh sang index trong tập từ điển Tiếng Anh
    list_en_train_index, list_en_train_word = word_to_index(x_train, en_dict)
    list_en_val_index, list_en_val_word = word_to_index(x_val, en_dict)
    # return encode sentences to index in VietNamese train/valid data
    # mã hoá các câu Tiếng Việt sang index trong tập từ điển Tiếng Việt
    list_vn_train_index, list_vn_train_word = word_to_index(y_train, vn_dict)
    list_vn_val_index, list_vn_val_word = word_to_index(y_val, vn_dict)
    # return list_en_train_index, list_en_val_index, list_vn_train_index, list_vn_val_index
    return list_en_train_index, list_vn_train_index, list_en_val_index, list_vn_val_index


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


def filter_sentences(x_train, x_val, y_train, y_val, en_dict, vn_dict):
    list_en_train_index, list_vn_train_index, list_en_val_index, list_vn_val_index = \
        encode_data(x_train, x_val, y_train, y_val, en_dict, vn_dict)
    # filter data train and data valid min len 3 and max len 100
    # Lấy các câu trong tập dữ liệu huấn luyện và trong tập dữ liệu kiểm tra có độ dài tối thiểu là 3 và tối đa là 100
    list_en_train_index, list_vn_train_index = filter_data(list_en_train_index, list_vn_train_index, min_len=3,
                                                           max_len=100)
    list_en_val_index, list_vn_val_index = filter_data(list_en_val_index, list_vn_val_index, min_len=3, max_len=100)
    list_en_train_index, list_vn_train_index = sort_data_length(list_en_train_index, list_vn_train_index)
    list_en_val_index, list_vn_val_index = sort_data_length(list_en_val_index, list_vn_val_index)

    return list_en_train_index, list_vn_train_index, list_en_val_index, list_vn_val_index


def save_dict(obj, name):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

'''
if __name__ == '__main__':
    list_en, list_vn = read_file(path)
    x_train, x_val, y_train, y_val = split_dataset(list_en, list_vn)
    en_dict, vn_dict = dict_laguage(x_train, y_train)
    save_dict(en_dict, "en_dictionary")
    save_dict(vn_dict, "vn_dictionary")
    list_en_train_index, list_vn_train_index, list_en_val_index, list_vn_val_index = \
        filter_sentences(x_train, x_val, y_train, y_val, en_dict, vn_dict)
    dict_train_val = dict()
    dict_train_val["list_en_train_index"], dict_train_val["list_vn_train_index"] = list_en_train_index, list_vn_train_index
    dict_train_val["list_en_val_index"], dict_train_val["list_vn_val_index"] = list_en_val_index, list_vn_val_index
    save_dict(dict_train_val, "dict_train_val")
    dict_train_val = load_dict("dict_train_val")
    print(len(dict_train_val['list_en_train_index']))
    print(len(dict_train_val['list_vn_train_index']))
'''