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
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, matthews_corrcoef, recall_score, f1_score, precision_score
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from model import TransHLA
import argparse




def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)





def get_entropy(probs):
    ent = -(probs.mean(0) * torch.log2(probs.mean(0) + 1e-12)
            ).sum(0, keepdim=True)
    return ent





def get_cond_entropy(probs):
    cond_ent = -(probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
    return cond_ent





def get_val_loss(logits, label, criterion,alpha = 0.04):
    loss = criterion(logits.view(-1, 2), label.view(-1))
    loss = (loss.float()).mean()
    loss = (loss - alpha).abs() + alpha
    logits = F.softmax(logits, dim=1)

    sum_loss = loss+get_entropy(logits)-get_cond_entropy(logits)
    return sum_loss[0]





def get_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, 2), label.view(-1))
    loss = (loss.float()).mean()
    loss = (loss - 0.04).abs() + 0.04
    return loss






def test_eval(test, test_labels, model):
    Result, _ = model(test)
    _, predicted = torch.max(Result, 1)
    correct = 0
    correct += (predicted == test_labels).sum().item()
    return 100*correct/Result.shape[0]






def addbatch(data, label, batchsize,shuffle=True):
    data = TensorDataset(data, label)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=shuffle)
    return data_loader





def save_model(model_dict, best_acc, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = '{}.pt'.format(save_prefix)
    save_path_pt = os.path.join(save_dir, filename)
    print('save_path_pt', save_path_pt)
    torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)
    print('Save Model Over: {}'.format(save_prefix))






def test_loader_eval(test, test_labels, batchsize, device, model):
    model.eval()
    correct = 0
    length = 0
    # predict_list=[]
    Result_list = []
    labels_list = []
    predicted_list = []
    representation_list = []
    test_loader = addbatch(test, test_labels, batchsize,shuffle=False)
    for step, (inputs, labels) in enumerate(test_loader):
        print(step)
        inputs.to(device)
        labels.to(device)
        labels_list.append(labels)
        Result, representation = model(inputs.to(device))
        # print(torch.max(Result))
        Result_list.append(np.array(Result.detach().cpu()))
        representation_list.append(np.array(representation.detach().cpu()))
        _, predicted = torch.max(Result, 1)
        predicted_list.append(np.array(predicted.cpu()))
        # predict_list.append(Result)
        correct += (predicted.to(device) == labels.to(device)).sum().item()
        length += len(labels)
    representation_list = np.concatenate(representation_list)
    Result_list = np.concatenate(Result_list)
    labels_list = np.concatenate(labels_list)
    predicted_list = np.concatenate(predicted_list)
    #auc_score = roc_auc_score(np.array(labels_list),Result_list[:,1])
    mcc = matthews_corrcoef(labels_list, predicted_list)
    f1 = f1_score(labels_list, predicted_list)
    recall = recall_score(labels_list, predicted_list)
    precision = precision_score(labels_list, predicted_list)
    model.train()
    return 100*correct/test_labels.shape[0], mcc, f1, recall, precision, Result_list, predicted_list, labels_list, representation_list






# def training(model, device, epochs, criterion, optimizer, traindata, test, test_labels,train_path,train_name,patience=5):
#     running_loss = 0
#     max_performance = 0
#     early_stop_counter = 0
#     best_model_state = None
#     stop_training = False

#     for epoch in tqdm(range(epochs), desc="Epochs"):
#         if stop_training:
#             break

#         for step, (inputs, labels) in enumerate(tqdm(traindata, desc="Training", leave=False)):
#             model.train()
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             model = model.to(device)
#             outputs, _ = model(inputs)
#             # loss = criterion(outputs,labels)
#             loss = get_val_loss(outputs, labels, criterion)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         acc, mcc, f1, recall, precision, result, labels,predict_label, representation = test_loader_eval(
#             test, test_labels, 128, device, model)
    
#         if acc > max_performance:
#             print("best_model_save")
#             save_model(model.state_dict(), acc,
#                        train_path, train_name)
#             max_performance = acc
#             early_stop_counter = 0
#             best_model_state = model.state_dict()

#         else:
#             early_stop_counter += 1
#             if early_stop_counter >= patience:
#                 print("Early stopping triggered. No improvement in validation accuracy for {} epochs.".format(
#                     patience))
#                 stop_training = True

#         print("epoch {}: validation_accuracy {:.3f}, mcc {:.3f}, f1 {:.3f}, recall {:.3f} precision{}".format(
#             epoch + 1, acc, mcc, f1, recall,precision))
#         running_loss = 0

#         if stop_training:
#             break

#     if best_model_state is not None:
#         model.load_state_dict(best_model_state)

#     return model



def training(model, device, epochs, criterion, optimizer, traindata, test, test_labels, patience=5):
    running_loss = 0
    max_performance = 0
    early_stop_counter = 0
    best_model_state = None
    stop_training = False

    for epoch in range(epochs):
        if stop_training:
            break

        # Create a single progress bar for the entire epoch
        with tqdm(total=len(traindata), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for inputs, labels in traindata:
                model.train()
                inputs = inputs.to(device)
                labels = labels.to(device)
                model = model.to(device)
                outputs, _ = model(inputs)
                # loss = criterion(outputs, labels)
                loss = get_val_loss(outputs, labels, criterion)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Update the progress bar
                pbar.update(1)

        acc, mcc, f1, recall, precision, result, labels, predict_label, representation = test_loader_eval(
            test, test_labels, 128, device, model)
    
        if acc > max_performance:
            print("best_model_save")
            save_model(model.state_dict(), acc, train_path, train_name)
            max_performance = acc
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered. No improvement in validation accuracy for {patience} epochs.")
                stop_training = True

        print(f"epoch {epoch + 1}: validation_accuracy {acc:.3f}, mcc {mcc:.3f}, f1 {f1:.3f}, recall {recall:.3f} precision {precision}")
        running_loss = 0

        if stop_training:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model






def plot_embedding(embedding, labels, plot_name):
    # Create a new figure window
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create a color list based on the label values
    colors = ['blue' if label == 0 else 'red' for label in labels]

    # Scatter plot
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=1)

    # Set the title and axis labels with increased font size
    ax.set_title(plot_name, fontsize=22)
    ax.set_xlabel('First Dim', fontsize=20)
    ax.set_ylabel('Second Dim', fontsize=20)

    # Adjust the figure layout to ensure labels and title are fully displayed
    fig.tight_layout()

    # Save the entire figure as a PDF file
    fig.savefig(plot_name + '.pdf', format='pdf', dpi=30)

    # Show the plot
    plt.show()






def remove_samples_with_length_greater_than(series, length):
    series_filtered = series[series.str.len() <= length]
    return series_filtered





def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on HLA data")

    # 单独定义路径参数
    parser.add_argument('--train_path', type=str, required=True,
                        help='The path to the training data CSV file.')
    parser.add_argument('--validation_path', type=str, required=True,
                        help='The path to the validation data CSV file.')
    # ... 定义其他命令行参数

    return parser.parse_args()



def get_train_config_HLA():
    # model parameters 
    parse = argparse.ArgumentParser(description='TransHLA')
    parse.add_argument('-max_len',type = int, default = 14,help = 'max_len')
    parse.add_argument('-max-len', type=int, default=14, help='max length of input sequences')
    parse.add_argument('-n_layers', type=int, default=6, help='number of encoder blocks')  # 3
    parse.add_argument('-n_head', type=int, default=8, help='number of head in multi-head attention')  # 8
    parse.add_argument('-d_model', type=int, default=1280, help='residue embedding dimension')  # 64
    parse.add_argument('-dim-feedforward', type=int, default=64, help='hidden layer dimension in feedforward layer')
    parse.add_argument('-cnn_padding_index', type=float, default=0, help='padding_index')
    parse.add_argument('-cnn_num_channel', type=int, default=256, help='cnn_num_channel')
    parse.add_argument('-cnn_kernel_size', type=int, default=3, help='cnn_kernel_size')
    parse.add_argument('-cnn_padding_size', type=int, default=1,help='cnn_padding_size')
    parse.add_argument('-cnn_stride', type=int, default=1,help='cnn_stride')
    parse.add_argument('-pooling_size', type=int, default=2,help='pooling_size')
    parse.add_argument('-region_embedding_size',type=int,default = 3,help='region_embedding_size')
    parse.add_argument('--train_path', type=str, required=True,
                        help='The path to the training data CSV file.')
    parse.add_argument('--validation_path', type=str, required=True,
                        help='The path to the validation data CSV file.')
    parse.add_argument('--model_path', type=str, required=True,
                        help='The path to the model storage.')
    parse.add_argument('--model_name', type=str, required=True,
                        help='The name of the model.')

    config = parse.parse_args()
    return config


def main():
    set_seed(10038)
    parameters = get_train_config_HLA()
    
    train = pd.read_csv(parameters.train_path, header=0)
    validation = pd.read_csv(parameters.validation_path, header=0)
   
    
    train = train.drop_duplicates(keep='first') 
    x_validation = validation.iloc[:, 0]
    x_train = train.iloc[:, 0]
   
     
    
    
    #if the type is HLA_II, please use 21 as the parameter
    
    train_label = train.iloc[:, 1]
    validation_label = validation.iloc[:, 1]

    train_label = torch.tensor(np.array(train_label, dtype='int64'))
    validation_label = torch.tensor(np.array(validation_label, dtype='int64'))
    
    print('load_pretrain_model')
    
    esm2, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    x_train = list(zip(range(len(x_train)), x_train))
    x_validation = list(zip(range(len(x_validation)), x_validation))
    _, _, x_train_encoding = batch_converter(x_train)
    _, _, x_validation_encoding = batch_converter(x_validation)
    
    
    
    
    device = 'cuda'
    
    print('parameters')
    model = TransHLA(parameters)
   
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.00003, weight_decay=0.0025)
    criterion = torch.nn.CrossEntropyLoss()
    traindata = addbatch(x_train_encoding, train_label, 64)
    print('training_start')
    training(model, device, 100, criterion, optimizer,
              traindata, x_validation_encoding, validation_label,parameters.model_path,parameters.model_name)




if __name__ == "__main__":
    main()
