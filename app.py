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
