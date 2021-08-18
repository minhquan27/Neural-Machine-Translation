import numpy as np
import torch
from model import *
import pickle
from padding_data import *
from load_data import *

# path
path = "/Users/nguyenquan/Desktop/mars_project/neural_machine_translation/data/train_en2vi/"
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Experiment:
    def __init__(self, learning_rate=0.001, num_epochs=100, encoder_embedding_size=300,
                 decoder_embedding_size=300, hidden_size=1024, num_layers=2, drop_out=0.5, batch_size=32):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.encoder_embedding_size = encoder_embedding_size
        self.decoder_embedding_size = decoder_embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        self.batch_size = batch_size

    def load_pickle(self, name):
        with open(path + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def evaluate(self, model, data_src, data_target):
        pass

    def translate_sentences(self, model, sentences, english_vocab, vietnamese_vocab, max_length):
        list_w_to_index = [2]
        for word in sentences.split():
            word = word.lower()
            if preprocess_string(word) in english_vocab.keys():
                list_w_to_index.append(english_vocab[preprocess_string(word)])
            if preprocess_string(word) not in english_vocab.keys():
                list_w_to_index.append(english_vocab['<unk>'])
        list_w_to_index.append(3)
        sentences_tensor = torch.LongTensor(list_w_to_index).unsqueeze(1).to(device)
        with torch.no_grad():
            hidden, cell = model.Encoder_LSTM(sentences_tensor)
        outputs = [2]
        for _ in range(max_length):
            previous_word = torch.LongTensor([outputs[-1]]).to(device)

            with torch.no_grad():
                output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
                best_guess = output.argmax(1).item()

            outputs.append(best_guess)
            if output.argmax(1).item() == 3:
                break
        vietnamese_vocab = {v: k for k, v in vietnamese_vocab.items()}
        translated_sentence = [vietnamese_vocab[idx] for idx in outputs]
        return translated_sentence

    def bleu(self, model, english_vocab, vietnamese_vocab):
        pass

    def train_and_eval(self):
        en_dictionary = self.load_pickle("en_dictionary")
        vn_dictionary = self.load_pickle("vn_dictionary")
        dict_train_val = self.load_pickle("dict_train_val")
        list_en_train_index, list_vn_train_index = list(dict_train_val['list_en_train_index']), \
                                                   list(dict_train_val['list_vn_train_index'])
        list_en_val_index, list_vn_val_index = list(dict_train_val['list_en_val_index']), \
                                               list(dict_train_val['list_vn_val_index'])

        encoder_lstm = EncoderLSTM(len(en_dictionary), self.encoder_embedding_size,
                                   self.hidden_size, self.num_layers, self.drop_out).to(device)
        decoder_lstm = DecoderLSTM(len(vn_dictionary), self.decoder_embedding_size,
                                   self.hidden_size, self.num_layers, self.drop_out, len(vn_dictionary)).to(device)
        model = Seq2Seq(encoder_lstm, decoder_lstm).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        sentences_1 = "I go to school by bus"
        print("Starting training...")
        for it in range(1, self.num_epochs + 1):
            model.eval()
            translated_sentence1 = self.translate_sentences(model, sentences_1, en_dictionary, vn_dictionary, 50)
            print("Translate sentence 1: {}".format(translated_sentence1))
            model.train()
            losses = []
            for j in range(0, 1000, self.batch_size):
                src_batch_en, target_batch_vn = get_batch(j, list_en_train_index, list_vn_train_index, self.batch_size)
                src_batch_en = torch.Tensor(src_batch_en).to(torch.int64).T
                target_batch_vn = torch.Tensor(target_batch_vn).to(torch.int64).T
                # print(src_batch_en.shape)
                # print(target_batch_vn.shape)
                # Pass the input and target for model's forward method
                output = model(src_batch_en, target_batch_vn)
                output = output[1:].reshape(-1, output.shape[2])
                target = target_batch_vn[1:].reshape(-1)
                # Clear the accumulating gradients
                opt.zero_grad()
                # Calculate the loss value for every epoch
                loss = criterion(output, target)
                # Calculate the gradients for weights & biases using back-propagation
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print("Epoch: {}".format(it))
            print("Loss: {}".format(np.mean(losses)))
            model.eval()
            # with torch.no_grad():
            #    print("Validation:")
            #     self.evaluate(model, list_en_val_index, list_vn_val_index)


if __name__ == '__main__':
    experiment = Experiment(learning_rate=0.001, num_epochs=100, encoder_embedding_size=300,
                            decoder_embedding_size=300, hidden_size=1024, num_layers=2, drop_out=0.5, batch_size=100)
    experiment.train_and_eval()
