import re 
import time 
import os 
import gc 
import glob
import json
import torch 
import random 
import datetime
import tokenizers 
import numpy as np 
import transformers 
import pandas as pd 
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from tokenizers import * 
from transformers import * 
from functools import partial 
from tqdm.notebook import tqdm 
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import GroupKFold

#Initial Configurations
SEED = 2020
DATA_PATH = '/Users/amirshahcheraghian/Leveraging-BERT-for-Advanced-Question-Answering-Systems/Data/Coleridge Initiative - Show US the Data/'
DATA_PATH_TRAIN = DATA_PATH + 'train/'
DATA_PATH_TEST = DATA_PATH + 'test/'

NUM_WORKERS = 4
VOCABS = {"bert-base-uncased":"/Users/amirshahcheraghian/Leveraging-BERT-for-Advanced-Question-Answering-Systems/Data/Bert Vocabs/bert-base-uncased-vocab.txt",}

# Import the models and save model paths 
from transformers import AutoTokenizer, AutoModel

# Define your model paths
model_paths = {
    'bert-base-uncased': '/Users/amirshahcheraghian/BERT Models/bert-base-uncased/',
    'bert-large-uncased-whole-word-masking-finetuned-squad': '/Users/amirshahcheraghian/BERT Models/bert-large-uncased-whole-word-masking-finetuned-squad/',
    'albert-large-v2': '/Users/amirshahcheraghian/BERT Models/albert-large-v2/',
    'albert-base-v2': '/Users/amirshahcheraghian/BERT Models/albert-base-v2/'
}
#'distilbert': '/Users/amirshahcheraghian/BERT Models/distilbert/'
# Function to download and save model
def download_and_save_model(model_name, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

# Download and save each model
for model_name, path in model_paths.items():
    download_and_save_model(model_name, path)
    print(f"Model {model_name} saved to {path}")
