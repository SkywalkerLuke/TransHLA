
# TransHLA: A Hybrid Transformer Model for HLA-Presented Epitope Detection

This repository has open-sourced the source code of `TransHLA`, which is used for the comparative model, dataset, and the model training and inference process.

This document contains a tutorial on the model's training process, and it also includes the use of the `TransHLA` pre-trained model in transformers.

`TransHLA` is a tool designed to discern whether a peptide will be recognized by HLA as an epitope.`TransHLA` is the first tool capable of directly identifying peptides as epitopes without the need for inputting HLA alleles. Due the different length of epitopes, we trained two models. The first is TransHLA_I, which is used for the detection of the HLA-I epitope, the other is TransHLA_II, which is used for the detection of the HLA-II epitope. This document is the code and the datasets 


## Model description
   `TransHLA` is a hybrid transformer model that utilizes a transformer encoder module and a deep CNN module. It is trained using pretrained sequence embeddings from `ESM2` and contact map structural features as inputs. It can serve as a preliminary screening for the currently popular tools that are specific for HLA-epitope binding affinity.

## Intended uses

Due to variations in peptide lengths, our `TransHLA` is divided into `TransHLA_I` and `TransHLA_II`, which are used to separately identify epitopes presented by HLA class I and class II molecules, respectively. Specifically, `TransHLA_I` is designed for shorter peptides ranging from 8 to 14 amino acids in length, while `TransHLA_II` targets longer peptides with lengths of 13 to 21 amino acids. The output consists of three columns: the first column lists the peptide, the second column provides the probability of being an epitope, and the third column contains the predicted label, where 1 indicates an epitope and 0 indicates a non-epitope. 

Here is how to use the TransHLA model:

First, download this repository.
```
git clone https://github.com/SkywalkerLuke/TransHLA.git
cd TransHLA
```
Then, install the `requirements.txt`

```
pip install -r  requirements.txt
```
If you want to use the model in an isolated environment, we also provide the `Dockerfile`, `TransHLA.def`, `transhla_env.yaml`

We provide the `TransHLA_I.py` and `TransHLA_II.py` for user to use our model. TransHLA_I.py is used for predicting HLA Class I epitopes. TransHLA_II.py is used for predicting Class II epitopes. Here is how to use these files to predict whether a peptide is an epitope.

```
python TransHLA_I.py --test_path  --outputs_path 
```

```
python TransHLA_II.py --test_path  --outputs_path 
```

And the Example is added in the <a href="https://colab.research.google.com/drive/1snAqZTG9BxSVcvDzZA9ipgSWEPBSAJ3r?usp=sharing">Colab Example</a>.

### How to train your own model
First, download this repository.
```
git clone https://github.com/SkywalkerLuke/TransHLA.git
cd TransHLA
```
Then, install the `requirements.txt`

```
pip install -r  requirements.txt
```

Then, change the directory to the `model_train_test`, and use the `train.py`:

```
cd model_train_test
```
```
python train.py --train_path your_train.csv --validation_path your_validation.csv --model_path your_path_to_save_model --model_name your_model_name.pt
```

And you can use the `inference.py` to use your own model:

```
python inference.py --test_path your_test.csv --model_path your_model.pt --ouputs_path your_outputs_path.npy
```

And we add an example on colab of how to use `train.py` and `inference.py`：<a href="https://colab.research.google.com/drive/1snAqZTG9BxSVcvDzZA9ipgSWEPBSAJ3r?usp=sharing">Colab Example</a>.

### How to use in transformers
First, users need to download the following packages: `pytorch`, `fair-esm`, and `transformers`. Additionally, the CUDA version must be 11.8 or higher; otherwise, the model will need to be run on CPU.
``` 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install fair-esm
```
Here is how to use TransHLA_I model to predict whether a peptide is an epitope:

```python
from transformers import AutoTokenizer
from transformers import AutoModel
import torch



def pad_inner_lists_to_length(outer_list,target_length=16):
    for inner_list in outer_list:
        padding_length = target_length - len(inner_list)
        if padding_length > 0:
            inner_list.extend([1] * padding_length)
    return outer_list


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("SkywalkerLu/TransHLA_I", trust_remote_code=True)
    model.to(device)
    peptide_examples = ['EDSAIVTPSR','SVWEPAKAKYVFR']
    peptide_encoding = tokenizer(peptide_examples)['input_ids']
    peptide_encoding = pad_inner_lists_to_length(peptide_encoding)
    print(peptide_encoding)
    peptide_encoding = torch.tensor(peptide_encoding)
    outputs,representations = model(peptide_encoding.to(device))
    print(outputs)
    print(representations)
```
And here is how to use TransHLA_II model to predict the peptide whether epitope:

```python
from transformers import AutoTokenizer
from transformers import AutoModel
import torch




def pad_inner_lists_to_length(outer_list,target_length=23):
    for inner_list in outer_list:
        padding_length = target_length - len(inner_list)
        if padding_length > 0:
            inner_list.extend([1] * padding_length)
    return outer_list



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("SkywalkerLu/TransHLA_II", trust_remote_code=True)
    model.to(device)
    model.eval()
    peptide_examples = ['KMIYSYSSHAASSL','ARGDFFRATSRLTTDFG']
    peptide_encoding = tokenizer(peptide_examples)['input_ids']
    peptide_encoding = pad_inner_lists_to_length(peptide_encoding)
    peptide_encoding = torch.tensor(peptide_encoding)
    outputs,representations = model(peptide_encoding.to(device))
    print(outputs)
    print(representations)

```


