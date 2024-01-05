from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

from train import set_seed
from tokenizing import tokenized_dataset

def inference(model, tokenized_sent, device):
  '''
  test dataset을 DataLoader로 만들어 준 후,
  batch_size로 나눠 model이 예측한다.

  Args:
      model (class 'transformers.modeling_auto.AutoModelForSequenceClassification'): AutoModelForSequenceClassification에 해당하는 모델
      tokenized_sent (class 'RE_Dataset'): RE_Dataset 클래스 객체
      device (class 'torch.device'): PyTorch에서 디바이스를 나타내는 클래스

  Returns:
      List: 예측 결과에 대한 label(num) 값
      List: 각 label에 대한 확률 값
  '''
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for _, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  '''
  dict를 이용해 숫자로 된 class -> 문자열 label로 반환

  Args:
      label (_type_): _description_

  Returns:
      List: 문자열로 되어있는 label list 반환
  '''

  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

### load하는 부분은 train과 똑같으니 하나의 코드로 작성?
def load_test_dataset(dataset_dir, tokenizer):
  '''
  test dataset을 불러온 후, tokenizing 진행한다.
  

  Args:
      dataset_dir (String): test dataset directory
      tokenizer (_type_): 모델에 해당하는 tokenizer

  Returns:
      class 'pandas.core.series.Series': test_dataset['id']
      tensor: tokenized_test로 raw test 데이터를 토큰화한 데이터(preprocessing X)
      List: test_label로 현재 모든 test 데이터가 100으로 설정
  '''
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def main(args):
  '''
  주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드
  '''

  set_seed(42)
  
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # load tokenizer
  Tokenizer_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  ## load my model
  # MODEL_NAME = args.model_dir # model dir.
  model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
  model.to(device)

  ## load test datset
  test_dataset_dir = "../dataset/test/test_data.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
  output.to_csv('./prediction/submission.csv', index=False)

  print('---- Finish! ----')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./best_model")
  args = parser.parse_args()
  print(args)
  main(args)
  
