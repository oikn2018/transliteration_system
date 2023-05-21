# !pip install torchtext==0.6.0
# import locale
# locale.getpreferredencoding = lambda: "UTF-8"
# ! pip install wget
# ! pip install gdown
# ! pip install --upgrade gdown

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
from torch.autograd import Variable
import os
import csv
import gdown
from tqdm import tqdm
# import wandb
from io import open
import string, time, math
import wget
from zipfile import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
from torch.utils.data import Dataset
import re
from torchtext.data import Field, TabularDataset, BucketIterator
import numpy as np
import random
import argparse

# Seed to maximally ensure reproducibility of results as much as possible
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# CUDA
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


# Getting the Dataset
url = 'https://drive.google.com/uc?id=1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw&export=download'

if not os.path.exists("aksharantar_sampled"):
  filename = gdown.download(url = url, quiet=False, fuzzy=True)
  print(filename)
  with ZipFile(filename, 'r') as z:
    print('Extracting files...')
    z.extractall()
    print('Done!')
  os.remove(filename)

# Define the English alphabet and padding character
eng_alpha = 'abcdefghijklmnopqrstuvwxyz'
pad_char = '<PAD>'

# Create a mapping from English alphabet characters to their corresponding indices
eng_alpha2idx = {pad_char: 0}
for index, alpha in enumerate(eng_alpha):
  eng_alpha2idx[alpha] = index + 1

# Define the minimum and maximum Unicode hex ranges for the Indic language
min_range = 2304
max_range = 2431

# Choose indic lang from config
indic_lang = config['indic_lang']

# Specify the Unicode hex ranges for specific Indic languages (Bengali and Hindi)
if indic_lang == 'ben':
  min_range = 2432
  max_range = 2558
elif indic_lang == 'hin':
  min_range = 2304
  max_range = 2431

# Create a list of Indic alphabet characters within the specified Unicode hex range
indic_alpha = [chr(alpha) for alpha in range(min_range, max_range + 1)]

# Create a mapping from Indic alphabet characters to their corresponding indices
indic_alpha2idx = {pad_char: 0}
for index, alpha in enumerate(indic_alpha):
  indic_alpha2idx[alpha] = index + 1

# Create a reverse mapping from Indic alphabet indices to their corresponding characters
indic_idx2alpha = {v: k for k, v in indic_alpha2idx.items()}

# Create a reverse mapping from English alphabet indices to their corresponding characters
eng_idx2alpha = {v: k for k, v in eng_alpha2idx.items()}

# Tokenize a string into a list of indices for the Indic language
def tokenize_indic(string):
  char_list = [*string]
  char_list = [indic_alpha2idx[char] for char in char_list]
  return char_list

# Tokenize a string into a list of indices for the English language
def tokenize_eng(string):
  char_list = [*string]
  char_list = [eng_alpha2idx[char] for char in char_list]
  return char_list

import pandas as pd
  
file_names = ['test', 'train', 'valid']

for index, file_name in enumerate(file_names):
  # read contents of csv file
  file = pd.read_csv(f'aksharantar_sampled/{indic_lang}/{indic_lang}_{file_name}.csv')
    
  # adding header
  headerList = ['eng', f'{indic_lang}']
    
  # converting data frame to csv
  file.to_csv(f'aksharantar_sampled/{indic_lang}/{indic_lang}_{file_name}.csv', header=headerList, index=False)

# Define the Field objects for English and Indic languages
eng = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, init_token='<sos>', eos_token='<eos>')
indic = Field(sequential=True, use_vocab=True, tokenize=tokenize_indic, init_token='<sos>', eos_token='<eos>')

# Define the fields dictionary to specify field names and corresponding Field objects
fields={'eng': ('eng', eng), f'{indic_lang}': ('indic', indic)}

# Set the path to the directory containing the data files
path_name = f'aksharantar_sampled/{indic_lang}'

# Specify the file names for the training, validation, and test datasets
train_name = f'{indic_lang}_train.csv'
val_name = f'{indic_lang}_valid.csv'
test_name = f'{indic_lang}_test.csv'

# Load the data from the CSV files using TabularDataset.splits()
train_data, val_data, test_data = TabularDataset.splits(
    path= path_name,
    train=train_name,
    validation=val_name,
    test=test_name,
    format='csv',
    fields=fields
)

# Build the vocabulary for the English language
eng.build_vocab(train_data, max_size=1000, min_freq=1)

# Build the vocabulary for the Indic language
indic.build_vocab(train_data, max_size=1000, min_freq=1)

class Encoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, config=None):
    super(Encoder, self).__init__()

    # Check if configuration is provided, otherwise use default values
    if config is None:
      config = {
        "load_model": False,
        "test_model": False,
        "indic_lang": 'ben',
        "wandb_project": 'CS6910_Assignment3',
        "wandb_entity": 'dl_research',
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "batch_size": 128,
        "input_embedding_size": 256,
        "num_layers": 4,
        "hidden_size": 512,
        "cell_type": 'GRU',
        "bidirectional": False,
        "epochs": 30
      }

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = nn.Dropout(p)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.cell_type = config['cell_type']
    
    # Select the appropriate cell type based on the configuration
    if self.cell_type == 'LSTM':
      self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    elif self.cell_type == 'GRU':
      self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p)
    elif self.cell_type == 'RNN':
      self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=p)

  def forward(self, x):

    embedding = self.dropout(self.embedding(x))
    # Embedding shape: (seq_length, N, embedding_size)

    if self.cell_type == 'LSTM':
      outputs, (hidden, cell) = self.lstm(embedding)
      return hidden, cell
    elif self.cell_type == 'RNN':
      outputs, hidden = self.rnn(embedding)
      return hidden
    elif self.cell_type == 'GRU':
      outputs, hidden = self.gru(embedding)
      return hidden

    
class Decoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p, config=None):
    super(Decoder, self).__init__()

    # Check if configuration is provided, otherwise use default values
    if config is None:
      config = {
        "load_model": False,
        "test_model": False,
        "indic_lang": 'ben',
        "wandb_project": 'CS6910_Assignment3',
        "wandb_entity": 'dl_research',
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "batch_size": 128,
        "input_embedding_size": 256,
        "num_layers": 4,
        "hidden_size": 512,
        "cell_type": 'GRU',
        "bidirectional": False,
        "epochs": 30
      }

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.cell_type = config['cell_type']
    self.dropout = nn.Dropout(p)
    self.embedding = nn.Embedding(input_size, embedding_size)
    
    # Select the appropriate cell type based on the configuration
    if self.cell_type == 'LSTM':
      self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    elif self.cell_type == 'GRU':
      self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p)
    elif self.cell_type == 'RNN':
      self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=p)

    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden, cell):

    x = x.unsqueeze(0)
    # x shape: (1, N) to predict one word at a time for N words in a batch.

    embedding = self.dropout(self.embedding(x))
    # embedding shape: (1, N, embedding_size)

    if self.cell_type == 'LSTM':
      outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
    elif self.cell_type == 'GRU':
      outputs, hidden = self.gru(embedding, hidden)
    elif self.cell_type == 'RNN':
      outputs, hidden = self.rnn(embedding, hidden)
      
    # outputs shape: (1, N, hidden_size)
    predictions = self.fc(outputs)
    # predictions shape: (1, N, length_of_vocab) -> (N, length_of_vocab)
    predictions = predictions.squeeze(0)

    if self.cell_type == 'LSTM':
      return predictions, hidden, cell
    else:
      return predictions, hidden


class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, config=None):
    super(Seq2Seq, self).__init__()

    if config is None:
      config = {
        "load_model": False,
        "test_model": False,
        "indic_lang": 'ben',
        "wandb_project": 'CS6910_Assignment3',
        "wandb_entity": 'dl_research',
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "batch_size": 128,
        "input_embedding_size": 256,
        "num_layers": 4,
        "hidden_size": 512,
        "cell_type": 'GRU',
        "bidirectional": False,
        "epochs": 30
      }

    self.run_name = 'cell_{}_do_{}_lr_{}_bs_{}_iem_{}_nl_{}_hs_{}_bidir_{}_ep_{}'.format(config['cell_type'], config['dropout'], config['learning_rate'], config['batch_size'], config['input_embedding_size'], config['num_layers'], config['hidden_size'], config['bidirectional'], config['epochs'])
    self.encoder = encoder
    self.decoder = decoder
    self.cell_type = config['cell_type']

  def forward(self, source, target, teacher_force_ratio=0.5):
    batch_size = source.shape[1]  # source dim: (target_len, N) -> N: batch size
    target_len = target.shape[0]
    target_vocab_size = len(indic.vocab)

    # predict 1 word at a time, but do it for an entire batch, every vector will be of that entire vocab size
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
    if self.cell_type == 'LSTM':
      hidden, cell = self.encoder(source)
    else:
      hidden = self.encoder(source)

    # Grab start token
    x = target[0]

    # send to decoder word by word
    for t in range(1, target_len):
      if self.cell_type == 'LSTM':
        output, hidden, cell = self.decoder(x, hidden, cell)
      else:
        output, hidden = self.decoder(x, hidden, cell=None)

      outputs[t] = output  # adding along 1st dimension -> target_len
      # output dim -> (N, english_vocab_size) -> doing argmax along this dimension, we'll get index corresponding to best guess that decoder outputted.
      best_guess = output.argmax(1)

      # implementing ground truth
      x = target[t] if random.random() < teacher_force_ratio else best_guess

    return outputs
def translit_infer(model, word, eng, indic, device, max_length=50, config=None):
    tokens = tokenize_eng(word)

    # Add <SOS> and <EOS> in the beginning and end respectively
    tokens.insert(0, eng.init_token)
    tokens.append(eng.eos_token)

    text_to_indices = [eng.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    word_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
      if config['cell_type'] == 'LSTM':
        hidden, cell = model.encoder(word_tensor)
      else:
        hidden = model.encoder(word_tensor)

    outputs = [indic.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_char = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
          if config['cell_type'] == 'LSTM':
            output, hidden, cell = model.decoder(previous_char, hidden, cell)
          else:
            output, hidden = model.decoder(previous_char, hidden, cell=None)
          best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == indic.vocab.stoi["<eos>"]:
            break

    translit_res = [indic.vocab.itos[idx] for idx in outputs]

    # remove start token
    translit_res_word = ''
    translit_res = translit_res[1:]

    for i in translit_res:
      if i != "<eos>":
        translit_res_word += indic_idx2alpha[i]
      else:
        break

    return translit_res_word



model_name=""
# Saving Checkpoint Code
def save_checkpoint(state, filename=f"{indic_lang}_{model_name}_checkpoint.pth.tar"):
    print("-x- Saving checkpoint -x-")
    torch.save(state, filename)

# Loading Checkpoint Code
def load_checkpoint(checkpoint, model, optimizer):
    print("-x- Loading checkpoint -x-")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("-x- Checkpoint Loaded Successfully! -x-")
def check_accuracy(loader, model, input_shape=None, toggle_eval=True, print_test=False, config=None):
    if toggle_eval:
        model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device
    num_correct = 0
    num_samples = 0
    print_list = []  # List to store rows for printing predictions
    with torch.no_grad():
        loader.create_batches()  # Create batches from the loader
        for batch in loader.batches:  # Iterate over batches
          for example in batch:  # Iterate over examples in the batch
            print_row_dict = {}  # Dictionary to store row values for printing predictions
            num_samples += 1
            eng_word = "".join([eng_idx2alpha[val] for val in example.eng])  # Convert English indices to characters and join them to form the word
            indic_word = "".join([indic_idx2alpha[val2] for val2 in example.indic])  # Convert Indic indices to characters and join them to form the word
            indic_pred = translit_infer(model, eng_word, eng, indic, device, max_length=50, config=config)  # Perform transliteration inference on the model
            if config['test_model'] and print_test:
              # Store values in the print row dictionary for printing predictions
              print_row_dict['english_word'] = eng_word
              print_row_dict['ground_truth'] = indic_word
              print_row_dict['predicted_word'] = indic_pred
              print_list.append(print_row_dict)  # Append the row dictionary to the print list
            if indic_pred == indic_word:  # Check if the predicted word matches the ground truth word
              num_correct += 1
    if config['test_model'] and print_test:
      fields = ["english_word", "ground_truth", "predicted_word"]

      # Write the print list to a CSV file for printing predictions
      with open('predictions_vanilla.csv', 'w', newline='') as file:
          writer = csv.DictWriter(file, fieldnames=fields)
          writer.writeheader()
          writer.writerows(print_list)

    accuracy = num_correct / num_samples  # Calculate the accuracy
    if toggle_eval:
        model.train()  # Set the model back to training mode
    return accuracy

# Determine the device to use for training/testing
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device being used to train/test model: {torch.cuda.get_device_name(0)}')

sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'dropout': {
            'values': [0.3, 0.5, 0.7]
        },
        'learning_rate': {
            'values': [0.001,0.005,0.0001,0.0005]
        },
        'batch_size': {
            'values': [64, 128, 256]
        },
        'input_embedding_size': {
            'values': [256, 512]
        },
        'num_layers': {
            'values': [1, 2, 3, 4]
        },
        'hidden_size':{
            'values': [256, 512, 1024]
        },
        'cell_type': {
            'values': ['RNN', 'LSTM', 'GRU']
        },
        'bidirectional': {
            # 'values': [True, False]
            'values': [False]
        },
        # 'decoding_strategy': {
        #     'values': ['beam_search', 'greedy']
        # },
        # 'beam_sizes':{
        #     'values': [3, 5, 7]
        # },
        'epochs':{
            'values': [20, 25, 30]
            # 'values': [5]
        }
    }
}

sweep_id = wandb.sweep(sweep_config,project='CS6910_Assignment3', entity='dl_research')
# sweep_id = wandb.sweep(sweep_config,project=config['wandb_project'], entity=config['wandb_entity'])

def train():
    torch.cuda.empty_cache()  # Clear GPU cache

    with wandb.init() as run:
        config = wandb.config

        # Training Hyperparameters
        num_epochs = config['epochs']
        learning_rate = config['learning_rate']
        batch_size = config['batch_size']

        # Model Hyperparameters
        load_model = False
        input_size_encoder = len(eng.vocab)
        input_size_decoder = len(indic.vocab)
        output_size = len(indic.vocab)
        encoder_embedding_size = config['input_embedding_size']
        decoder_embedding_size = config['input_embedding_size']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        enc_dropout = config['dropout']
        dec_dropout = config['dropout']
        cell_type = config['cell_type']
        bidirectional = config['bidirectional']

        train_iterator, val_iterator, test_iterator = BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size=batch_size,
            # Examples of similar length will be in the same batch to minimize padding and save on compute
            sort_within_batch=True,
            sort_key=lambda x: len(x.eng),
            device=device)

        encoder_net = Encoder(
            input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout, config=config
        ).to(device)
        decoder_net = Decoder(
            input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout, config=config
        ).to(device)

        model = Seq2Seq(encoder_net, decoder_net, config=config).to(device)
        model_name = model.run_name
        run.name = model_name
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        pad_idx = indic.vocab.stoi['<pad>']
        # If all examples in a batch are of similar length, don't incur a penalty for padding
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        if load_model:
            load_checkpoint(torch.load(f'{indic_lang}_{model_name}_checkpoint.pth.tar'), model, optimizer)

        if indic_lang == 'hin':
            word = 'bachta'
            og_translit = 'बचता'
        elif indic_lang == 'ben':
            word = 'gabhaaskar'
            og_translit = 'গাভাস্কার'
        acc_val_prev = 0
        acc_val_current = 0

        print(f'Hyperparameter settings: {model_name}')
        for epoch in range(num_epochs):
            print(f'Epoch [{epoch + 1} / {num_epochs}]')

            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            if acc_val_current > acc_val_prev:
                if os.path.exists(f'{indic_lang}_{model_name}_checkpoint.pth.tar'):
                    os.remove(f'{indic_lang}_{model_name}_checkpoint.pth.tar')
                acc_val_prev = acc_val_current
                save_checkpoint(checkpoint, f'{indic_lang}_{model_name}_checkpoint.pth.tar')

            loop = tqdm(enumerate(train_iterator), total=len(train_iterator))
            for batch_idx, batch in loop:
                inp_data = batch.eng.to(device)
                target = batch.indic.to(device)

                output = model(inp_data, target)
                # output shape: (target_len, batch_size, output_dim)

                # Reshape output keeping the last output_dim the same
                output = output[1:].reshape(-1, output.shape[2])  # So that the first start token is not sent to our model
                # target shape: (target_len, batch_size)
                target = target[1:].reshape(-1)
                optimizer.zero_grad()
                loss = criterion(output, target)

                loss.backward()

                # To avoid exploding gradients, clip them when they are above a threshold
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

            model.eval()  # Turn off Dropout
            translit_res = translit_infer(model, word, eng, indic, device, max_length=50, config=config)
            print(f'Translated example word:  English: {word}, Actual: {og_translit}, Predicted: {translit_res}')
            model.train()

            print('Computing Loss and Validation Accuracy...')
            acc_val_current = check_accuracy(val_iterator, model, input_shape=None, toggle_eval=True, print_accuracy=True, config=config)

            metrics = {
                "loss": loss.item(),
                "val_accuracy": acc_val_current,
                "epochs": epoch
            }

            wandb.log(metrics)

            print(f'Training Loss: {loss.item()}, Validation Accuracy: {acc_val_current * 100:.2f}%')
            print('--------------------------')

wandb.agent(sweep_id, function=train, project='CS6910_Assignment3', entity='dl_research', count=5)
