import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # from keras.backend.tensorflow_backend import set_session
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import scipy.io as sio
# import matlab.engine
import torch

from utils import *
from data_preprocessing.CRBP.getSequence import *
from data_preprocessing.CRBP.BertDealEmbedding import *
from data_preprocessing.CRBP.getCircRNA2Vec import *


def get_data(protein):

    Kmer, dataY = dealwithdata1(protein) # X
    Embedding = dealwithCircRNA2Vec(protein)
    Embedding1 = circRNABert(protein, 3)

    np.random.seed(4)
    indexes = np.random.choice(Kmer.shape[0], Kmer.shape[0], replace=False)

    training_idx, test_idx = indexes[:round(((Kmer.shape[0])/10)*8)], indexes[round(((Kmer.shape[0])/10)*8):] #8:2
    X_train_1, X_test_1 = Kmer[training_idx, :, :], Kmer[test_idx, :, :]
    X_train_2, X_test_2 = Embedding[training_idx, :, :], Embedding[test_idx, :, :]
    X_train_3, X_test_3 = Embedding1[training_idx, :, :], Embedding1[test_idx, :, :] #(892,101,24)
    y_train, y_test = dataY[training_idx], dataY[test_idx]

    train_dataset = dict()
    train_dataset["samples1"] = torch.from_numpy(X_train_1)
    train_dataset["samples2"] = torch.from_numpy(X_train_2)
    train_dataset["samples3"] = torch.from_numpy(X_train_3)
    train_dataset["labels"] = torch.from_numpy(y_train)

    test_dataset = dict()
    test_dataset["samples1"] = torch.from_numpy(X_test_1)
    test_dataset["samples2"] = torch.from_numpy(X_test_2)
    test_dataset["samples3"] = torch.from_numpy(X_test_3)
    test_dataset["labels"] = torch.from_numpy(y_test)

    torch.save(train_dataset,'data/{}_train.pt'.format(protein))
    torch.save(test_dataset,'data/{}_test.pt'.format(protein))

    return train_dataset, test_dataset






