import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from konlpy.tag import Okt

import gensim
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

from dataset import Wusinsa
from models import MultiHeadAttention, TransformerBlock, TokenAndPositionEmbedding

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten, GRU
from keras.utils import np_utils


## training Bi-LSTM model
def Bilstm_model(embedding_vectors, maxlen):
    model = Sequential()
    model.add(Embedding(embedding_vectors.shape[0],
                        output_dim=embedding_vectors.shape[1],
                        weights=[embedding_vectors],
                        input_length=maxlen,
                        trainable=False))
    model.add(Bidirectional(LSTM(256, dropout = 0.5, return_sequences=True)))
    model.add(Bidirectional(LSTM(128,  return_sequences = False)))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model


## SSAN training
def ssan(maxlen, vocab_size):

    embedding_dim = 32  # 각 단어의 임베딩 벡터의 차원
    num_heads = 2  # 어텐션 헤드의 수
    dff = 32  # 포지션 와이즈 피드 포워드 신경망의 은닉층의 크기

    inputs = tf.keras.layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embedding_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embedding_dim, num_heads, dff)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    print(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--embed', type=int, default="200", help='Input the embedding size of model')
    parser.add_argument('--windows', type=int, default="3", help='determining the length of window for Word2vec model')
    parser.add_argument('--maxlen', type=int, default='50', help='determining the maximum length of sentence input')
    parser.add_argument('--epochs', type=int, default='10', help='determining the number of epochs')
    parser.add_argument('--batch', type=int, default='100', help='determining the number of batch size')
    parser.add_argument('--model', type=str, default='lstm', help='determining the kind of model')

    args = parser.parse_args()

    train = pd.read_csv("./wusinsa_prep.csv")

    embedding_size = args.embed
    windows = args.windows
    max_len = args.maxlen

    dataset = Wusinsa(embedding_size, windows, max_len, train)
    x_rvs_train, x_rvs_test, y_train, y_test, x_meta_train, x_meta_test = dataset.data_setting()
    X_train, X_test = dataset.preprocessing(x_rvs_train, x_rvs_test)
    trn_tkns, test_tkns, w2v, vocab = dataset.word2vec(X_train, X_test)
    embedding_vectors = dataset.w2v_to_keras_weights(w2v, vocab)
    trn_emb, test_emb = dataset.word_embeddings(X_train, X_test)

    if args.model == "bilstm":
        model = Bilstm_model(embedding_vectors=embedding_vectors, maxlen=max_len)
        model.summary()
        es = keras.callbacks.EarlyStopping(monitor="val_acc", patience=3, restore_best_weights=True)
        history = model.fit(trn_tkns, y_train, epochs=args.epochs, validation_split=0.2, batch_size=args.batch, verbose=1, callbacks=[es])

        print("\n 테스트 정확도: %.4f" % (model.evaluate(test_tkns, y_test)[1]))

        PATH = './models/wusinsa_bilstm.pth'
        model.save(PATH)

    elif args.model == "ssan":

        model = ssan(args.maxlen, len(vocab)+2)

        history = model.fit(trn_emb, y_train, batch_size=args.batch, epochs=args.epochs, validation_split=0.2)
        # history = model.fit(trn_tkns, y_train, batch_size=32, epochs=2, validation_data=(text_val_tok_pad, val_Y))
        print("\n 테스트 정확도: %.4f" % (model.evaluate(test_tkns, y_test)[1]))

    else:
        print("Bert Not yet")