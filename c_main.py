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

from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate

## training Bi-LSTM model
def Bilstm_model(embedding_vectors, maxlen):

    nlp_input = Input(shape=(maxlen,), name='nlp_input')
    meta_input = Input(shape=(7,), name='meta_input')
    emb = Embedding(output_dim=embedding_vectors.shape[1], input_dim=embedding_vectors.shape[0],
                    weights=[embedding_vectors], input_length=maxlen)(nlp_input)
    nlp_out = Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01)
                                 , return_sequences=True))(emb)
    nlp_out = Bidirectional(LSTM(128, return_sequences=False))(nlp_out)
    x = concatenate([nlp_out, meta_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[nlp_input, meta_input], outputs=[x])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model


## SSAN training
def ssan(maxlen, vocab_size):

    embedding_dim = 32  # 각 단어의 임베딩 벡터의 차원
    num_heads = 2  # 어텐션 헤드의 수
    dff = 32  # 포지션 와이즈 피드 포워드 신경망의 은닉층의 크기

    inputs = tf.keras.layers.Input(shape=(maxlen,))
    meta_inputs = tf.keras.layers.Input(shape=(7,), name='meta_input')
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embedding_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embedding_dim, num_heads, dff)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.concatenate([x, meta_inputs])
    x = tf.keras.layers.Dense(30, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs=[inputs, meta_inputs], outputs=outputs)
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

    m_train = dataset.meta_onehot(x_meta_train)
    m_test = dataset.meta_onehot(x_meta_test)

    ## add category informations
    meta_train = np_utils.to_categorical(m_train['meta'])
    meta_test = np_utils.to_categorical(m_test['meta'])

    X_train, X_test = dataset.preprocessing(x_rvs_train, x_rvs_test)
    ## Bi-LSTM
    trn_tkns, test_tkns, w2v, vocab = dataset.word2vec(X_train, X_test)
    embedding_vectors = dataset.w2v_to_keras_weights(w2v, vocab)
    ## SSAN
    trn_emb, test_emb = dataset.word_embeddings(X_train, X_test)

    if args.model == "bilstm":
        model = Bilstm_model(embedding_vectors=embedding_vectors, maxlen=max_len)
        model.summary()
        es = keras.callbacks.EarlyStopping(monitor="val_acc", patience=2, restore_best_weights=True)
        history = model.fit([trn_tkns, meta_train], y_train, epochs=args.epochs, validation_split=0.2, batch_size=args.batch, verbose=1, callbacks=[es])

        print("\n 테스트 정확도: %.4f" % (model.evaluate([test_tkns, meta_test], y_test)[1]))

    elif args.model == "ssan":

        model = ssan(args.maxlen, len(vocab)+2)

        history = model.fit([trn_emb,meta_train], y_train, batch_size=args.batch, epochs=args.epochs, validation_split=0.2)
        # history = model.fit(trn_tkns, y_train, batch_size=32, epochs=2, validation_data=(text_val_tok_pad, val_Y))
        print("\n 테스트 정확도: %.4f" % (model.evaluate([test_tkns, meta_test], y_test)[1]))

    else:
        print("Bert Not yet")