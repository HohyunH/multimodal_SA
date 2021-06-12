import numpy as np
import pandas as pd
from tqdm import tqdm
from konlpy.tag import Okt

import gensim
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing import sequence
from keras.preprocessing.text import Tokenizer

class Wusinsa():

    def __init__(self, embedding_size, windows, maxlen, df):
        self.embedding_size = embedding_size
        self.windows = windows
        self.maxlen = maxlen
        self.df = df

    ## split dataset
    def data_setting(self):
        using_df = self.df[['categories','rvs','meta_sizes','meta_brights','meta_colors','meta_thicks', 'scores',
                            "cus_sex", "cus_height","cus_weight"]]
        binary = []
        for score in using_df['scores']:
            if score == 'score10':
                binary.append(0)
            else:
                binary.append(1)

        using_df['binary'] = binary

        using_df['rvs'] = using_df['rvs'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

        x_train, x_test, y_train, y_test = train_test_split(using_df[['categories','rvs','meta_sizes','meta_brights','meta_colors',
                                                                        'meta_thicks', "cus_sex", "cus_height","cus_weight"]],
                                                            using_df['binary'], test_size=0.10, random_state=321)

        x_rvs_train = x_train['rvs'].reset_index()['rvs']
        x_meta_train = x_train.reset_index()[['categories','meta_sizes','meta_brights','meta_colors', 'meta_thicks', "cus_sex", "cus_height","cus_weight"]]

        x_rvs_test = x_test['rvs'].reset_index()['rvs']
        x_meta_test = x_test.reset_index()[['categories', 'meta_sizes', 'meta_brights', 'meta_colors', 'meta_thicks', "cus_sex", "cus_height", "cus_weight"]]

        return x_rvs_train, x_rvs_test, np.array(y_train), np.array(y_test), x_meta_train, x_meta_test

    ## Tokenizing review
    def preprocessing(self, x_rvs_train, x_rvs_test):

        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다', '대다',
                     '년', '월', '대']

        okt = Okt()

        X_train = []
        for sentence in tqdm(x_rvs_train):
            temp_X = []
            temp_X = okt.morphs(sentence, stem=True)  # 토큰화
            temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
            X_train.append(temp_X)

        X_test = []
        for sentence in tqdm(x_rvs_test):
            temp_X = []
            temp_X = okt.morphs(sentence, stem=True)  # 토큰화
            temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
            X_test.append(temp_X)

        return X_train, X_test

    ## for Bi-LSTM
    def word2vec(self, X_train, X_test):

        w2v_model = gensim.models.Word2Vec(sentences=X_train,
                                           vector_size=self.embedding_size,
                                           window=self.windows,
                                           min_count=1)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)
        text_train_tok = tokenizer.texts_to_sequences(X_train)
        word_index = tokenizer.word_index
        print('Sive of vocabulary: ', len(word_index))

        text_train_tok_pad = pad_sequences(text_train_tok, maxlen=self.maxlen)

        text_test_tok = tokenizer.texts_to_sequences(X_test)
        text_test_tok_pad = pad_sequences(text_test_tok, maxlen=self.maxlen)

        return text_train_tok_pad, text_test_tok_pad, w2v_model, word_index

    def w2v_to_keras_weights(self, model, vocab):
        vocab_size = len(vocab) + 1
        weight_matrix = np.zeros((vocab_size, self.embedding_size))
        for word, i in vocab.items():
            weight_matrix[i] = model.wv[word]
        return weight_matrix

    ## for SSAN
    def word_embeddings(self, X_train, X_test):

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)
        total_cnt = len(tokenizer.word_index)
        vocab_size = total_cnt + 2
        tokenizer = Tokenizer(vocab_size, oov_token='OOV')
        tokenizer.fit_on_texts(X_train)
        text_train_tok = tokenizer.texts_to_sequences(X_train)
        word_index = tokenizer.word_index
        print('Sive of vocabulary: ', len(word_index))
        text_train_tok_pad = sequence.pad_sequences(text_train_tok, maxlen=self.maxlen, value=0)

        text_test_tok = tokenizer.texts_to_sequences(X_test)
        text_test_tok_pad = sequence.pad_sequences(text_test_tok, maxlen=self.maxlen, value=0)

        return text_train_tok_pad, text_test_tok_pad

if __name__ == "__main__":

    train = pd.read_csv("./wusinsa_prep.csv")
    print(train.shape)

    dataset = Wusinsa(200, 3, 50, train)

    x_rvs_train, x_rvs_test, y_train, y_test, x_meta_train, x_meta_test = dataset.data_setting()

    X_train, X_test = dataset.preprocessing(x_rvs_train, x_rvs_test)

    trn_tkns, test_tkns, w2v, vocab = dataset.word2vec(X_train, X_test)

    embedding_vectors = dataset.w2v_to_keras_weights(w2v, vocab)

    trn_emb, test_emb = dataset.word_embeddings(X_train, X_test)

