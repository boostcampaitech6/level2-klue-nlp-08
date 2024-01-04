import pickle as pickle
import os
import pandas as pd
import torch


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  
  # pair_dataset, labels를 입력으로 받아옴
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  # data label의 길이 반환
  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1] # i는 subject entity의 word의 value
    j = j[1:-1].split(',')[0].split(':')[1] # j는 object entity의 word의 value

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  """
  input: 0,〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.,"{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}","{'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'}",no_relation,wikipedia
  output: {'id': 0, 'sentence': '〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.', 'subject_entity':'비틀즈', 'object_entity':'조지 해리슨', 'label':'no relation'}
  """
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir) # csv 읽어와서 Dataframe으로 변경
  dataset = preprocessing_dataset(pd_dataset) # preprocessing 진행 -> output은 Dataframe
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = '' # 이건 왜 있지???
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer( ### 여기 알아보기
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt", # retun type = tensor
      padding=True, # max_length 보다 짧은 문장 padding
      truncation=True, # max_length 보다 긴 문장 자르기
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences
