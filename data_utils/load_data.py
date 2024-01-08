import pickle as pickle
# import os
import pandas as pd
import torch

from preprocessing.preprocessing import *

class RE_Dataset(torch.utils.data.Dataset):
  '''
  train.py에서 Trainer 인스턴스의 dataset으로 사용될 RE_Dataset 클래스 정의
  '''
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


# dataset csv파일 경로
def load_data(dataset_dir):
  '''
  csv 파일을 경로에 맡게 불러와 전처리를 진행한다.

  Args:
      dataset_dir (String): 데이터 csv 파일 경로

  Returns:
      DataFrame: 전처리된 데이터셋을 반환(토큰화 X)
  '''
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset