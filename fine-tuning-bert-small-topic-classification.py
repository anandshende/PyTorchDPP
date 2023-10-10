import os
import nltk
import torch
import numpy as np
import pandas as pd
import tokenizers
import transformers
from spacy.lang.en import English
import nltk
import subprocess
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import torch
import tqdm
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

tqdm.tqdm.pandas()

def read_csv():
    train_df = pd.read_csv('/home/ubuntu/Desktop/Anand/Research/notebooks/datasets/ag-news-classification/train.csv')
    test_df = pd.read_csv('/home/ubuntu/Desktop/Anand/Research/notebooks/datasets/ag-news-classification/test.csv')
    return train_df, test_df
    
def remove_stop_words(train_df: pd.DataFrame):
    if os.path.isfile('./removed_stop_words.csv'):
        filtered_train_df = pd.read_csv('removed_stop_words.csv')
        return filtered_train_df

    nlp = English()
    lemmatizer = WordNetLemmatizer()
    def stemandstop(row):
        new_title = []
        title = row.Title
        my_doc = nlp(title)
        for i in my_doc:
            if i.is_stop is False:
                new_title.append(lemmatizer.lemmatize(i.text))
        row['Title'] = ' '.join(new_title)
        
        new_description = []
        description = row.Description
        my_doc = nlp(description)
        for i in my_doc:
            if i.is_stop is False:
                new_description.append(lemmatizer.lemmatize(i.text))
        row['Description'] = ' '.join(new_description)

        return row

    filtered_train_df = train_df.progress_apply(stemandstop, axis=1)
    filtered_train_df.to_csv('./removed_stop_words.csv')
    return filtered_train_df

def convert_to_word_tokens(filtered_train_df):
    if os.path.isfile('./after_converting_to_word_tokens.csv'):
        filtered_train_df = pd.read_csv('after_converting_to_word_tokens.csv')
        return filtered_train_df
    
    from gensim.utils import simple_preprocess
    try:
        # print(filtered_train_df.Title.head())
        filtered_train_df.Title = filtered_train_df.Title.fillna('').astype(str)
        filtered_train_df.Description = filtered_train_df.Description.fillna('').astype(str)
        filtered_train_df.Title = filtered_train_df.Title.progress_apply(simple_preprocess, min_len=3)
        filtered_train_df.Description = filtered_train_df.Description.progress_apply(simple_preprocess, min_len=3)
        filtered_train_df['Title'] = filtered_train_df['Title'].map(lambda x: ' '.join(x))
        filtered_train_df['Description'] = filtered_train_df['Description'].map(lambda x: ' '.join(x))
        filtered_train_df['combined'] = filtered_train_df['Title'].str.cat(filtered_train_df['Description'])
        filtered_train_df['target'] = torch.nn.Sigmoid()(torch.tensor(filtered_train_df['Class Index']))
        filtered_train_df = filtered_train_df.sample(frac=1).reset_index(drop=True)
        filtered_train_df.to_csv('./after_converting_to_word_tokens.csv')
    except Exception as e:
        print('\n\n\n\n Exceptions')
        print(e)
        
    return filtered_train_df

class TopicDataset(Dataset):
  def __init__(self, X, y):
    self.x_train = X
    self.y_train = y
    
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self, idx):
    return self.x_train[idx], self.y_train[idx]

def form_a_dataset(input_df):
    topicDataset = TopicDataset(input_df["combined"], input_df["target"])
    return DataLoader(topicDataset, batch_size=2, sampler=DistributedSampler(topicDataset))
    
def start_fine_tuning():

    class BertSmallClassifier(torch.nn.Module):
        def __init__(self, dropout_rate=0.3):
            super(BertSmallClassifier, self).__init__()
            
            self.roberta = BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
            self.d1 = torch.nn.Dropout(dropout_rate)
            self.flatten = torch.nn.Flatten()
            self.l1 = torch.nn.Linear(128, 16)
            self.bn1 = torch.nn.LayerNorm(16)
            self.d2 = torch.nn.Dropout(dropout_rate)
            self.l2 = torch.nn.Linear(16, 1)
            self.sigmoid = torch.nn.Sigmoid()
            
        def forward(self, inputs):
            _, out = self.roberta(**inputs, return_dict=False)
            # print(out)
            x = out
            x = self.d1(x)
            x = self.l1(x)
            x = self.bn1(x)
            x = torch.nn.Tanh()(x)
            x = self.d2(x)
            x = self.l2(x)
            x = self.sigmoid(x)
            return x
        
    model = BertSmallClassifier()
    return model

def run_epochs(model, train_loader, epochs_run):
    tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100
    save_every = 10
    loss_arr = []
    train_loader.sampler.set_epoch(epochs)
    for i in tqdm.tqdm(range(epochs_run, epochs)):
        for index, data in enumerate(train_loader):
            y_train = data[-1]
            tokenized_input = tokenizer(
                list(data[0]),
                return_tensors='pt',
                max_length=150,
                padding='max_length',
                truncation=True
            )
        
            optimizer.zero_grad()
            
            # print('\n\n\n')
            # print(tokenized_input)
            
            y_hat = model.forward(
                tokenized_input
                # input_ids=torch.tensor(tokenized_input['input_ids'].squeeze(1)),
                # attention_mask=torch.tensor(tokenized_input['attention_mask'])
            )
            loss = criterion(y_hat, y_train.float().unsqueeze(1))
            loss_arr.append(loss)

            loss.backward()
            optimizer.step()

        if i % save_every == 0:
            print(f'Epoch: {i} Loss: {loss_arr[-1]}')
            _save_snapshot(model, i)
            
            
def ddp_setup(rank: int, world_size: int):
#   """
#   Args:
#       rank: Unique identifier of each process
#      world_size: Total number of processes
#   """
#   os.environ["MASTER_ADDR"] = "localhost"
#   os.environ["MASTER_PORT"] = "12355"
#   init_process_group(backend="nccl", rank=rank, world_size=world_size)
    init_process_group(backend="cpu:gloo")
#   torch.cuda.set_device(rank)

def _save_snapshot(model, epoch):
    snapshot = {}
    snapshot["MODEL_STATE"] = model.module.state_dict()
    snapshot["EPOCHS_RUN"] = epoch
    torch.save(snapshot, "BertSmallClassifier.pt")
    print(f"Epoch {epoch} | Training snapshot saved at BertSmallClassifier.pt")
    
def _load_snapshot(model, snapshot_path):
    snapshot = torch.load(snapshot_path)
    model.load_state_dict(snapshot["MODEL_STATE"])
    epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Resuming training from snapshot at Epoch {epochs_run}")
    return epochs_run

print('before starting')
ddp_setup(0, 2)
print('after ddp setup')
train_df, test_df = read_csv()
print('after reading csv')
filtered_train_df = remove_stop_words(train_df)
print('after removing stop words')
filtered_train_df = convert_to_word_tokens(filtered_train_df)
print('convert to word tokens')
data_loader = form_a_dataset(filtered_train_df)
print('after data loaders')
model = start_fine_tuning()
print('after model creation')
model = DDP(model)
print('after ddp model creation')

if os.path.exists('BertSmallClassifier.pt'):
    epochs_run = _load_snapshot('BertSmallClassifier.pt')
    run_epochs(model, data_loader, epochs_run)
else:
    run_epochs(model, data_loader, 0)
print('after running epochs')