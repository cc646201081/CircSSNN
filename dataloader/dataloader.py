import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data_preprocessing.CRBP.getDataView import get_data
import os
import numpy as np


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        X_data1 = dataset["samples1"]
        X_data2 = dataset["samples2"]
        X_data3 = dataset["samples3"]

        y_data = dataset["labels"]

        self.x_data1 = X_data1
        self.x_data2 = X_data2
        self.x_data3 = X_data3
        self.y_data = y_data.long()
        self.len = X_data1.shape[0]


    def __getitem__(self, index):

        return self.x_data1[index], self.x_data2[index], self.x_data3[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(protein, configs):
    try:
        train_dataset = torch.load("../CircSSNN/data/{}_train.pt".format(protein))#train
        test_dataset = torch.load("../CircSSNN/data/{}_test.pt".format(protein)) #test
    except:
        print("{}数据集加载出错，重新生成！".format(protein))
        train_dataset, test_dataset = get_data(protein)

    train_dataset = Load_Dataset(train_dataset)
    test_dataset = Load_Dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=True, # 去掉末尾不够batch_size的样本configs.drop_last
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=True, drop_last=True, # 去掉末尾不够batch_size的样本configs.drop_last
                                              num_workers=0)

    return train_loader, test_loader
