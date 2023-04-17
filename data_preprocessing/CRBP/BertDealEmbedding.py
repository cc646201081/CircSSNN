import os
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" #!!
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
import pickle
from tqdm.auto import tqdm
import tensorflow as tf

def read_fasta1(file_path):
    seq_list = []
    f = open(file_path,'r')
    for line in f:
        if '>' not in line:
            line = line.strip().upper()
            seq_list.append(line)
    return seq_list

def seq2kmer1(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def circRNA_Bert(sequences, dataloader):
    features = []
    seq = []    
    tokenizer = BertTokenizer.from_pretrained('./3-new-12w-0/', do_lower_case=False)
    model = BertModel.from_pretrained("./3-new-12w-0/")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model = model.eval()
    for sequences in dataloader:
        seq.append(sequences)
    
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        token_type_ids = torch.tensor(ids['token_type_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)[0]
        embedding = embedding.cpu().numpy()
    
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd) 
    return features
    


def circRNABert(protein,k):
    file_positive_path = './Datasets/circRNA-RBP/' + protein + '/positive'
    file_negative_path = './Datasets/circRNA-RBP/' + protein + '/negative'
    sequences_pos = read_fasta1(file_positive_path)
    sequences_neg = read_fasta1(file_negative_path)
    sequences_ALL = sequences_pos + sequences_neg
    sequences = []
    Bert_Feature = []  
    for seq in sequences_ALL:
        seq = seq.strip()
        seq_parser = seq2kmer1(seq, k)
        sequences.append(seq_parser)

    dataloader = torch.utils.data.DataLoader(sequences, batch_size=100, shuffle=False)

    Features = circRNA_Bert(sequences, dataloader)

    for i in Features:
        Feature = np.array(i)
        Bert_Feature.append(Feature.tolist())
    arrayBF = np.array(Bert_Feature)
    data = np.pad(arrayBF, ((0,0),(0,2),(0,0)), 'constant', constant_values=0)
    return  data
