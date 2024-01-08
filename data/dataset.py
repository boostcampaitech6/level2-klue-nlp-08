import torch
import pandas as pd
import pickle as pickle
from transformers import AutoTokenizer

from utils.utils import label_to_num
from preprocessing.preprocessing import preprocessing_dataset
from preprocessing.tokenizing import tokenized_dataset

class RE_Dataset(torch.utils.data.Dataset):
    '''
    output : {input_ids:[tensor], attention_mask:[tensor], label:[tensor], type_id:[tensor]}
    '''

    def __init__(self, data_path, tokenizer_name):
        self.dataset = load_data(data_path)        
        self.pair_dataset = tokenized_dataset(
                                self.dataset,
                                AutoTokenizer.from_pretrained(tokenizer_name))

        self.labels = label_to_num(self.dataset['label'].values)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):

        return len(self.labels)


def load_data(dataset_dir):
  '''
  csv 파일을 읽고 전처리를 수행하여 dataframe으로 반환합니다.
  '''

  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset