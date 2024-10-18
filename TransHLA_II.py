import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score, confusion_matrix, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import seaborn as sns
from argparse import Namespace
import random
from transformers import AutoTokenizer, AutoModel

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


import argparse

def addbatch(data, label=None, batchsize=32, shuffle=True):
    if label is not None:
        dataset = TensorDataset(data, label)
    else:
        dataset = TensorDataset(data)
    
    data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
    return data_loader

def config():
    parse = argparse.ArgumentParser(description='TransHLA')
    parse.add_argument('--test_path', type=str, required=True, help='The path to the test data CSV file.')
    parse.add_argument('--outputs_path', type=str, required=True, help='The path of the outputs')
    config = parse.parse_args()
    return config

def pad_inner_lists_to_length(outer_list, target_length=23):
    for inner_list in outer_list:
        padding_length = target_length - len(inner_list)
        if padding_length > 0:
            inner_list.extend([1] * padding_length)
    return outer_list

def test_loader(test, batchsize, device, model):
    model.eval()
    Result_list = []
    predicted_list = []
    test_loader = addbatch(test, None, batchsize, shuffle=False)

    for inputs in test_loader:
        
        inputs = inputs[0]
        Result, _ = model(inputs)
        Result_list.append(np.array(Result.detach().cpu()))
        _, predicted = torch.max(Result, 1)
        predicted_list.append(np.array(predicted.cpu()))

    Result_list = np.concatenate(Result_list)
    predicted_list = np.concatenate(predicted_list)
    
    return Result_list, predicted_list

def main():
    parameters = config()
    test = pd.read_csv(parameters.test_path, header=0)
    outputs_path = parameters.outputs_path
    
    x_test = test.iloc[:, 0]
    x_test = x_test.reset_index(drop=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("SkywalkerLu/TransHLA_II", trust_remote_code=True)
    model.to(device)
    
    x_test = list(zip(range(len(x_test)), x_test))
    peptide_encoding = tokenizer([seq for _, seq in x_test])['input_ids']
    peptide_encoding = pad_inner_lists_to_length(peptide_encoding)
    peptide_encoding = torch.tensor(peptide_encoding)
    
    Result_list, predicted_list = test_loader(
        peptide_encoding.to(device), 128, device, model)
    
    probabilities = Result_list[:, 1]  # Assuming the second column is the probability of being class 1
    output_df = pd.DataFrame({
        'Peptide': [seq for _, seq in x_test],
        'Probability': probabilities,
        'Predicted Label': predicted_list
    })
    
    output_df.to_csv(outputs_path, index=False)
    
    print('Results saved to CSV')

if __name__ == "__main__":
    main()