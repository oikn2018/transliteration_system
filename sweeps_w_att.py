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

    if config is None:
      # Default configuration if not provided
      config = {
        "wandb_project": 'CS6910_Assignment3',
        "wandb_entity": 'dl_research',
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "batch_size": 256,
        "input_embedding_size": 256,
        "num_layers": 2,
        "hidden_size": 1024,
        "cell_type": 'GRU',
        "bidirectional": True,
        "epochs": 30
      }

    self.hidden_size = hidden_size
    self.bidirectional = config['bidirectional']
    self.num_layers = num_layers
    self.dropout = nn.Dropout(p)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.cell_type = config['cell_type']

    # Choose the appropriate recurrent cell type and configure it based on the provided parameters
    if self.cell_type == 'LSTM' and self.bidirectional:
      self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True, dropout=p)
    elif self.cell_type=='LSTM':
      self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    elif self.cell_type == 'GRU' and self.bidirectional:
      self.gru = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=True, dropout=p)
    elif self.cell_type == 'GRU':
      self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p)
    elif self.cell_type == 'RNN' and self.bidirectional:
      self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, bidirectional=True, dropout=p)
    elif self.cell_type == 'RNN':
      self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=p)


  def forward(self, x):

    # embedding shape: (seq_length, N, embedding_size)
    embedding = self.dropout(self.embedding(x))
    

    if self.cell_type == 'LSTM' and self.bidirectional:
      encoder_states, (hidden, cell) = self.lstm(embedding)
      hidden = (hidden[0:self.num_layers] + hidden[self.num_layers:2*self.num_layers])/2
      cell = (cell[0:self.num_layers] + cell[self.num_layers:2*self.num_layers])/2
      return encoder_states, hidden, cell
    
    elif self.cell_type == 'LSTM':
      encoder_states, (hidden, cell) = self.lstm(embedding)
      return encoder_states, hidden, cell
    
    elif self.cell_type == 'RNN' and self.bidirectional:
      encoder_states, hidden = self.rnn(embedding)
      hidden = (hidden[0:self.num_layers] + hidden[self.num_layers:2*self.num_layers])/2
      return encoder_states, hidden
    
    elif self.cell_type == 'RNN':
      encoder_states, hidden = self.rnn(embedding)
      return encoder_states, hidden
    
    elif self.cell_type == 'GRU' and self.bidirectional:
      encoder_states, hidden = self.gru(embedding)
      hidden = (hidden[0:self.num_layers] + hidden[self.num_layers:2*self.num_layers])/2
      return encoder_states, hidden
    
    elif self.cell_type == 'GRU':
      encoder_states, hidden = self.gru(embedding)
      return encoder_states, hidden

    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p, config=None):
      super(Decoder,self).__init__()

      if config is None:
        # Default configuration if not provided
        config = {
          "wandb_project": 'CS6910_Assignment3',
          "wandb_entity": 'dl_research',
          "dropout": 0.3,
          "learning_rate": 0.0005,
          "batch_size": 256,
          "input_embedding_size": 256,
          "num_layers": 2,
          "hidden_size": 1024,
          "cell_type": 'GRU',
          "bidirectional": True,
          "epochs": 30
        }
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.cell_type = config['cell_type']
      self.bidirectional = config['bidirectional']
      self.dropout = nn.Dropout(p)
      self.embedding = nn.Embedding(input_size, embedding_size)
      self.dir = 2 if self.bidirectional else 1

      if self.cell_type == 'LSTM':
        self.lstm = nn.LSTM(hidden_size*self.dir + embedding_size, hidden_size, num_layers, dropout=p)
      elif self.cell_type == 'GRU':
        self.gru = nn.GRU(hidden_size*self.dir + embedding_size, hidden_size, num_layers, dropout=p)
      elif self.cell_type == 'RNN':
        self.rnn = nn.RNN(hidden_size*self.dir + embedding_size, hidden_size, num_layers, dropout=p)

      # Linear layer to calculate attention scores
      self.energy = nn.Linear(hidden_size * (self.dir+1), 1)
      self.U = nn.Linear(hidden_size*self.dir, hidden_size)
      self.W = nn.Linear(hidden_size, hidden_size)
      self.attn = nn.Linear(hidden_size, 1)
      self.softmax = nn.Softmax(dim=0)
      self.relu = nn.ReLU()

      # Linear layer for final output prediction
      self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):

      # Expand dimensions to match the expected input shape
      x = x.unsqueeze(0)
      # Shape of x: (1, N)

      embedding = self.dropout(self.embedding(x))
      # embedding shape: (1,N, embedding_size)


      sequence_length = encoder_states.shape[0]
      # Implementation Logic 1: taking final layer hidden tensor value and passing it onto energy.
      hidden_temp = hidden[-1].unsqueeze(0)
      # Implementation Logic 2: taking average of all layer hidden tensor values and passing it onto energy.
      # hidden_temp = torch.mean(hidden, dim=0).unsqueeze(0)
      # Shape of hidden_temp: (1, N, hidden_size)

      # Using Baudanau Additive Attention Mechanism

      # Calculate energy scores for attention
      U = self.U(encoder_states)
      # Shape of U: (seq_length, N, hidden_size)

      # Expand the hidden states to match the sequence length
      W = self.W(hidden_temp.repeat(sequence_length, 1, 1))
      # Shape of W: (seq_length, N, hidden_size)

      energy = self.attn(torch.tanh(U+W))
#     # Shape of energy: (seq_length, N, 1)

      # Compute attention values
      attention = self.softmax(energy)
      # Shape of attention: (seq_length, N, 1)

      # Calculate the context vector using attention
      context_vector = torch.bmm(attention.permute(1, 2, 0), encoder_states.permute(1, 0, 2)).permute(1,0,2)
      # Shape of context_vector: (1, N, hidden_size)

      # Concatenate the context vector and embedded input
      rnn_input = torch.cat((context_vector, embedding), dim=2)
      # Shape of rnn_input: (1, N, hidden_size + embedding_size)

      # Pass the concatenated input through the LSTM/GRU/RNN
      if self.cell_type == 'LSTM':
        outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
      elif self.cell_type == 'GRU':
        outputs, hidden = self.gru(rnn_input, hidden)
      elif self.cell_type == 'RNN':
        outputs, hidden = self.rnn(rnn_input, hidden)
      
      # Final output prediction
      predictions = self.fc(outputs)
      # Shape of predictions: (1, N, output_size)

      # Remove the first dimension to match the target shape
      predictions = predictions.squeeze(0)
      # Shape of predictions: (N, output_size)

      if self.cell_type == 'LSTM':
        return predictions, hidden, cell
      else:
        return predictions, hidden


class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, config=None):
    super(Seq2Seq, self).__init__()

    if config is None:
      # Default configuration if not provided
      config = {
        "wandb_project": 'CS6910_Assignment3',
        "wandb_entity": 'dl_research',
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "batch_size": 256,
        "input_embedding_size": 256,
        "num_layers": 2,
        "hidden_size": 1024,
        "cell_type": 'GRU',
        "bidirectional": True,
        "epochs": 30
      }

    self.run_name = 'att_cell_{}_do_{}_lr_{}_bs_{}_iem_{}_nl_{}_hs_{}_bidir_{}_ep_{}'.format(config['cell_type'], config['dropout'], config['learning_rate'], config['batch_size'], config['input_embedding_size'], config['num_layers'], config['hidden_size'], config['bidirectional'], config['epochs'])
    self.encoder = encoder
    self.decoder = decoder
    self.cell_type = config['cell_type']

  def forward(self, source, target, teacher_force_ratio = 0.5):

    batch_size = source.shape[1]
    target_len = target.shape[0]
    target_vocab_size = len(indic.vocab)

    # Initialize outputs tensor
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
    
    # Encode the source sequence
    if self.cell_type == 'LSTM':
        encoder_states, hidden, cell = self.encoder(source)
    else:
        encoder_states, hidden = self.encoder(source)

    # Initialize the first target token
    x = target[0]

    # Generate output predictions word by word
    for t in range(1, target_len):
        if self.cell_type == 'LSTM':
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
        else:
            output, hidden = self.decoder(x, encoder_states, hidden, cell=None)

        outputs[t] = output  # Adding along the first dimension (target_len)
        
        # Select the next input based on teacher forcing ratio
        best_guess = output.argmax(1)

        # Implementing Teacher Forcing
        if random.random() >= teacher_force_ratio:
           x = best_guess
        else:
           x = target[t]

    return outputs

def translit_infer(model, word, eng, indic, device, max_length=50, config = None):
    tokens = tokenize_eng(word)

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens = [eng.init_token] + tokens + [eng.eos_token]

    # Convert tokens to indices
    text_to_indices = [eng.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    word_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
      if config['cell_type'] == 'LSTM':
        encoder_states, hidden, cell = model.encoder(word_tensor)
      else:
        encoder_states, hidden = model.encoder(word_tensor)

    outputs = [indic.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_char = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
          if config['cell_type'] == 'LSTM':
            output, hidden, cell = model.decoder(previous_char, encoder_states, hidden, cell)
          else:
            output, hidden = model.decoder(previous_char, encoder_states, hidden, cell=None)
          best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Check if the model predicts the end of the sentence
        if best_guess == indic.vocab.stoi["<eos>"]:
            break

    translit_res = [indic.vocab.itos[idx] for idx in outputs]

    # remove start token
    translit_res_word = ''
    translit_res = translit_res[1:]
    # return translit_res
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
        model.eval()
    
    device = next(model.parameters()).device
    num_correct = 0
    num_samples = 0
    print_list = []
    
    with torch.no_grad():
        loader.create_batches()
        for batch in loader.batches:
            for example in batch:
                print_row_dict = {}
                num_samples += 1
                eng_word = "".join([eng_idx2alpha[val] for val in example.eng])
                indic_word = "".join([indic_idx2alpha[val2] for val2 in example.indic])
                indic_pred = translit_infer(model, eng_word, eng, indic, device, max_length=50, config=config)
                
                if config['test_model'] and print_test:
                    # Store the details for printing the test results
                    print_row_dict['english_word'] = eng_word
                    print_row_dict['ground_truth'] = indic_word
                    print_row_dict['predicted_word'] = indic_pred
                    print_list.append(print_row_dict)
                
                if indic_pred == indic_word:
                    num_correct += 1
    
    if config['test_model'] and print_test:
        fields = ["english_word", "ground_truth", "predicted_word"]
        
        # Write the test results to a CSV file
        with open('predictions_attention.csv', 'w', newline='') as file: 
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(print_list)
    
    accuracy = num_correct / num_samples
    
    if toggle_eval:
        model.train()
    
    return accuracy


# Check if CUDA is available and set the device accordingly
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device being used to train/test model: {torch.cuda.get_device_name(0)}')

sweep_config_w_att = {
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
            'values': [True, False]
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

sweep_id_w_att = wandb.sweep(sweep_config_w_att,project='CS6910_Assignment3', entity='dl_research')


def train():
    torch.cuda.empty_cache()
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
        batch_size = batch_size,
        # Examples of similar length will be in same batch to minimize padding and save on compute
        sort_within_batch = True,
        sort_key = lambda x: len(x.eng),
        device = device)



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
        # if all examples in batch are of similar length, don't incur penalty for this padding
        criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

        if load_model:
          load_checkpoint(torch.load(f'{indic_lang}_{model_name}_checkpoint.pth.tar'), model, optimizer)

        if indic_lang == 'hin':
          word = 'bachta'
          og_translit = 'बचता'
        elif indic_lang == 'ben':
          word = 'stagecraft'
          og_translit = 'স্টেজক্রাফট'
        acc_val_prev = 0
        acc_val_current = 0
        epoch_loss = 0
        print(f'Hyperparameter settings: {model_name}')



        for epoch in range(num_epochs):
          print(f'Epoch [{epoch+1} / {num_epochs}]')

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

            #basically reshape output keeping last output_dim same
            output = output[1:].reshape(-1, output.shape[2]) # so that first start token is not sent to out model
            # target -> (target_len, batch_size)
            target = target[1:].reshape(-1)
            optimizer.zero_grad()
            loss = criterion(output, target)

            loss.backward()

            # to avoid exploding gradients, clip them when they are above a threshold
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            epoch_loss += loss.item()

          model.eval() # turns off Dropout
          translit_res = translit_infer(model, word, eng, indic, device, max_length=50, config=config)
          print(f'Translated example word:  English: {word}, Actual: {og_translit}, Predicted: {translit_res}')
          model.train()

          print('Computing Loss and Validation Accuracy...')
          acc_val_current = check_accuracy(val_iterator, model, input_shape=None, toggle_eval=True, print_accuracy=True, config=config)
          epoch_loss = epoch_loss/len(train_iterator)
          metrics = {
            "loss":epoch_loss,
            "val_accuracy": acc_val_current,
            "epochs":(epoch)
            }

          wandb.log(metrics)

          print(f'Training Loss: {epoch_loss}, Validation Accuracy: {acc_val_current * 100:.2f}%')
          print('--------------------------')
          epoch_loss = 0

wandb.agent(sweep_id_w_att, function=train,project='CS6910_Assignment3', entity='dl_research', count=15)

