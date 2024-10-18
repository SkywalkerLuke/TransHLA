import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
import collections
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
import random
from sklearn.metrics import auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#import esm
from tqdm import tqdm
import time
import seaborn as sns
from argparse import Namespace
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, matthews_corrcoef, recall_score, f1_score, precision_score
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from model import TransHLA
from train import addbatch,plot_embedding,test_loader_eval,remove_samples_with_length_greater_than
import argparse

def config():
    # model parameters 
    parse = argparse.ArgumentParser(description='TransHLA')
    parse.add_argument('-max_len',type = int, default = 14,help = 'max length of input sequences')
    parse.add_argument('-n_layers', type=int, default=6, help='number of encoder blocks')  # 3
    parse.add_argument('-n_head', type=int, default=8, help='number of head in multi-head attention')  # 8
    parse.add_argument('-d_model', type=int, default=1280, help='residue embedding dimension')  # 64
    parse.add_argument('-dim_feedforward', type=int, default=64, help='hidden layer dimension in feedforward layer')
    parse.add_argument('-cnn_padding_index', type=float, default=0, help='padding_index')
    parse.add_argument('-cnn_num_channel', type=int, default=256, help='cnn_num_channel')
    parse.add_argument('-cnn_kernel_size', type=int, default=3, help='cnn_kernel_size')
    parse.add_argument('-cnn_padding_size', type=int, default=1,help='cnn_padding_size')
    parse.add_argument('-cnn_stride', type=int, default=1,help='cnn_stride')
    parse.add_argument('-pooling_size', type=int, default=2,help='pooling_size')
    parse.add_argument('-region_embedding_size',type=int,default = 3,help='region_embedding_size')
    parse.add_argument('--test_path', type=str, required=True,
                        help='The path to the test data CSV file.')
    parse.add_argument('--model_path',  type=str, required=True,help = 'The path to the model pickle file')
    parse.add_argument('--outputs_path',  type=str, required=True,help = 'The path of the outputs')
    config = parse.parse_args()
    return config


def main():
    parameters = config()
    test = pd.read_csv(parameters.test_path, header=0)
    outputs_path = parameters.outputs_path
    max_len = parameters.max_len
    # test = pd.read_csv("../data/HLA_II_epitope_test.csv", header=0)
    
    # test = pd.read_csv("../data/HLA_I_external_1_time_negative.csv", header=0)
    # test = pd.read_csv("../data/HLA_II_external_1_time_negative.csv", header=0)
    # test = pd.read_csv("../data/HLA_I_external_4_time_negative.csv", header=0)
    # test = pd.read_csv("../data/HLA_II_external_4_time_negative.csv", header=0)
    
    
    x_test = test.iloc[:, 0]
    
    
    #if the type is HLA_II, please use 21 as the parameter
    x_test = remove_samples_with_length_greater_than(x_test, max_len)
    test_label = test.iloc[:, 1][x_test.index].reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    
    esm2, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    
    
    device = 'cuda'
    
    
    model = TransHLA(parameters)
    model.load_state_dict(torch.load(
        parameters.model_path))
    # model.load_state_dict(torch.load(
    #     '../II_model_save/TransHLA.pt'))
    model.eval()
    model.to(device)
    
    x_test = list(zip(range(len(x_test)), x_test))
    _, _, x_test_encoding = batch_converter(x_test)
    test_label = torch.tensor(np.array(test_label, dtype='int64'))
    
        
    acc,  mcc, f1, recall, precision, result, predict_label, labels, representation = test_loader_eval(
        x_test_encoding, test_label, 128, device, model)
    
    
    
    print('model_acc: '+ str(acc))
    print('f1_score: '+ str(f1))
    print('mcc: ' + str(mcc))
    print('recall: ' + str(recall))
    print('precision: ' + str(precision))
    print('save result')
    # outputs = np.save(predict_label)
    np.save(outputs_path,predict_label)
    #plot_auc
    # fpr, tpr, thresholds = roc_curve(np.array(labels), result[:, 1])
    # roc_auc = auc(fpr, tpr)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal', adjustable='box')
    
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc="lower right")
    # plt.show()
    
    
    
        
    # #plot_confusion_matrix
    # tn, fp, fn, tp = confusion_matrix(labels, predict_label).ravel()
    # specificity = tn / (tn + fp)
    
    
    
    # cm = confusion_matrix(labels, predict_label)
    
    # cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # fig, ax = plt.subplots()
    
    
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    
    
    
    
    
    
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
    #         ax.text(j + 0.5, i + 0.6, f'\n{cm_norm[i, j]*100:.1f}%',
    
    #                 ha='center', va='center', color=text_color)
    
    # ax.set_xlabel('Predicted')
    # ax.set_ylabel('True')
    # ax.set_title('TransHLA One-time negative External Confusion matrix')
    
    
    
    # tick_labels = ['Negative', 'Positive']
    # ax.xaxis.set_ticklabels(tick_labels)
    # ax.yaxis.set_ticklabels(tick_labels)
    # plt.show()
    
    # embedding_loader = addbatch(x_test_encoding,test_label,32,shuffle=False)
    # esm2.to(device)
    # seq_emb_list = []
    # structure_emb_list = []
    
    
    
    # for step, (inputs, labels) in enumerate(embedding_loader):
    #     print(step)
    #     inputs.to(device)
    #     labels.to(device)
    #     # labels_list.append(labels)
    #     results = esm2(inputs.to(device),repr_layers=[33], return_contacts=True)
    #     seq_emb = results["representations"][33]
    #     structure_emb = results["contacts"]
    #     seq_emb_list.append(np.array(seq_emb.detach().cpu()))
    #     structure_emb_list.append(np.array(structure_emb.detach().cpu()))
        
        
        
        
    # seq_emb_list = np.concatenate(seq_emb_list)
    # structure_emb_list = np.concatenate(structure_emb_list)
    # test_label = np.array(test_label)
    # seq_emb_list = np.mean(seq_emb_list,axis=1)
    # structure_emb_list = np.mean(structure_emb_list,axis=1)
    # pca = PCA(n_components=2)
    # embedding = pca.fit_transform(representation)
    # HLA_I_random_embedding = np.random.rand(184697, 1024)
    # # HLA_II_random_embedding = np.random.rand(130634, 1024)
    # HLA_I_random_ermbedding = pca.fit_transform(HLA_I_random_embedding)
    # # HLA_II_random_ermbedding = pca.fit_transform(HLA_II_random_embedding)
    # ESM_seq_embedding = pca.fit_transform(seq_emb_list)
    # ESM_structure_embedding = pca.fit_transform(structure_emb_list)
    # # plot_embedding(embedding, labels, 'HLA-II embedding from TransHLA')
    # plot_embedding(ESM_seq_embedding,test_label,'HLA-I embedding from ESM sequence feature')
    # plot_embedding(ESM_structure_embedding,test_label,'HLA-I embedding from ESM structure feature')
    # plot_embedding(HLA_I_random_embedding, labels, 'HLA-I embedding from Random Pattern')
    # plot_embedding(embedding, labels, 'HLA-I embedding from TransHLA')
    # plot_embedding(HLA_II_random_embedding, labels, 'HLA-II embedding from Random Pattern')



if __name__ == "__main__":
    main()
