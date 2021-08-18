import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import random

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class EncoderLSTM(torch.nn.Module):
    # class Encoder using Bidirectional LSTM
    # Pha mã hoá của kiến trúc seq2seq sử dụng kiến trúc của mạng LSTM 2 chiều
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.tag = True
        # embedding matrix shape [input_size, embedding_size]
        # Ma trận nhúng có kích thước bằng [kích thước từ điển tiếng Anh, độ dài vec-tơ nhúng]
        self.embedding = nn.Embedding(input_size, embedding_size)
        # lstm input [embedding_size, hidden_size, num_layers]
        # Mạng LSTM có đầu vào [độ dài vec-tơ nhúng, tầng ẩn, số LSTM xếp chồng =2]
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # forward LSTM
        # quá trình lan truyền thẳng
        # input x src English shape [seq_len, batch_size]
        # Ma trận x có đầu vào [độ dài của câu, số lượng câu]
        embedding = self.dropout(self.embedding(x))
        # embedding shape [seq_len, batch_size, embedding_size]
        # Ma trận nhúng Embedding có kích thước [độ dài câu, số lượng câu, kích thước nhúng]
        outputs, (hidden_state, cell_state) = self.LSTM(embedding)
        # output Encoder return hidden state and cell state shape [num_layer, batch_size, hidden_size]
        # Kết quả của pha mã hoá là hidden_state và cell_state với kích thước [số lớp LSTM, số lượng câu, tầng ẩn]
        return hidden_state, cell_state


class DecoderLSTM(torch.nn.Module):
    # class Decoder using Bidirectional LSTM
    # Pha giải mã của kiến trúc seq2seq sử dụng kiến trúc của mạng LSTM 2 chiều
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = nn.Dropout(p)
        # embedding matrix shape [input_size, embedding_size]
        # Ma trận nhúng có kích thước bằng [kích thước từ điển tiếng Việt, độ dài vec-tơ nhúng]
        self.embedding = nn.Embedding(input_size, embedding_size)
        # lstm input [embedding_size, hidden_size, num_layers]
        # Mạng LSTM có đầu vào [độ dài vec-tơ nhúng, tầng ẩn, số LSTM xếp chồng =2]
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        # linear network shape (hidden_size, output_size)
        # Mạng tuyến tính có kích thước [hidden_size, output_size]
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):
        # forward LSTM
        # x shape [1, batch_size]
        # vector x có kích thước [1, bath_size]
        x = x.unsqueeze(0)
        # embedding shape [1, batch_size, embedding_dim]
        # Ma trận nhúng embedding tương ứng có kích thước [1, số lượng câu, kích thước nhúng]
        embedding = self.dropout(self.embedding(x))
        # outputs shape [1, batch_size, hidden_size] and hs, cs shape [num_layer, batch_size, hidden_size]
        # đầu ra out_puts có kích thước [1, số lượng batch, kích thước tầng ẩn] và các trạng thái h_s, c_s
        outputs, (hidden_state, cell_state) = self.LSTM(embedding, (hidden_state, cell_state))
        # predictions shape [1, batch_size, output_size]
        # dự đoán đi qua mạng tuyến tính lan truyền thẳng được kích thước [số câu, kích thước đầu ra]
        predictions = self.fc(outputs)
        # predictions shape [batch_size, output_size]
        predictions = predictions.squeeze(0)

        return predictions, hidden_state, cell_state


class Seq2Seq(nn.Module):
    # Seq2Seq model with the Encoder-Decoder architecture
    def __init__(self, Encoder_LSTM, Decoder_LSTM):
        super(Seq2Seq, self).__init__()
        self.Encoder_LSTM = Encoder_LSTM
        self.Decoder_LSTM = Decoder_LSTM

    def forward(self, source, target, tfr=0.5):
        # Shape sentences source english : (10, 32) [(Sentence length English + some padding), Number of Sentences]
        batch_size = source.shape[1]
        # Shape sentences target vietnamese: (14, 32) [(Sentence length VietNamese + some padding), Number of Sentences]
        target_len = target.shape[0]
        target_vocab_size = 10004
        # Shape outputs (14, 32, 5766)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        # Shape --> (hs, cs) (2, 32, 1024) ,(2, 32, 1024) [num_layers, batch_size size, hidden_size]
        hidden_state, cell_state = self.Encoder_LSTM(source)
        # Shape of x (32 elements)
        x = target[0]  # Trigger token <SOS>
        for i in range(1, target_len):
            # Shape --> output (32, 5766)
            output, hidden_state, cell_state = self.Decoder_LSTM(x, hidden_state, cell_state)
            outputs[i] = output
            best_guess = output.argmax(1)  # 0th dimension is batch size, 1st dimension is word embedding
            # Either pass the next word correctly from the dataset or use the earlier predicted word
            x = target[i] if random.random() < tfr else best_guess
        # Shape --> outputs (14, 32, 5766)
        return outputs
