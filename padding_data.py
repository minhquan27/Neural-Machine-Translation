import numpy as np


def padding_src_en(sentences_src, seq_len):
    features_src = np.ones((len(sentences_src), seq_len), dtype=int)
    for ii, sentences in enumerate(sentences_src):
        features_src[ii, 0:len(sentences)] = np.array(sentences)[:seq_len]
        features_src[ii, len(sentences) - 1] = 1
        features_src[ii, -1] = 3
    return features_src


def padding_target_vn(sentences_tag, seq_len):
    # padding for y_train list sequences VietNamese
    # Thêm đệm cho các câu dữ liệu Tiếng Việt
    features_tag = np.ones((len(sentences_tag), seq_len), dtype=int)
    for ii, sentences in enumerate(sentences_tag):
        features_tag[ii, 0:len(sentences)] = np.array(sentences)[:seq_len]
    return features_tag


def get_batch(index, src_en, target_vn, batch_size):
        batch_src = src_en[index: index + batch_size]
        batch_target = target_vn[index: index + batch_size]
        max_seq_len_src = len(batch_src[-1])
        max_seq_len_target = max([len(i) for i in batch_target])
        batch_src = padding_src_en(batch_src, max_seq_len_src)
        batch_target = padding_target_vn(batch_target, max_seq_len_target)
        return batch_src, batch_target

"""
if __name__ == '__main__':
    '''
    x = [[2, 1684, 3], [2, 2527, 435, 133, 3], [2, 784, 3], [2, 251, 3], [2, 337, 11, 1171, 1014, 3]]
    print(len(x))
    for i in range(0, len(x), 3):
        print("lan", i)
        a = x[i:i + 3]
        print(a)
        max_seq_len = max([len(i) for i in a])
        tag = padding_target_vn(a, max_seq_len)
        print(tag)
    '''
    '''
    x = [[2, 930, 3], [2, 6988, 3], [2, 2228, 6, 7, 3], [2, 593, 3], [2, 6225, 3]]
    print(len(x))
    for i in range(0, len(x), 3):
        print("lan", i)
        a = x[i:i + 3]
        print(a)
        max_seq_len = len(a[-1])
        tag = padding_src_en(a, max_seq_len)
        print(tag)
   '''
    x_scr = [[2, 930, 3], [2, 6988, 3], [2, 2228, 6, 7, 3], [2, 593, 3], [2, 6225, 3]]
    x_target = [[2, 1684, 3], [2, 2527, 435, 133, 3], [2, 784, 3], [2, 251, 3], [2, 337, 11, 1171, 1014, 3]]
    create_train_loader(x_scr, x_target, 3)
"""