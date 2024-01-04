import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
import numpy as np
import random
from data_loader import load_and_tokenize_data
from metrics import compute_metrics
from model import load_model

def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train():
  set_seed(42)
  # 모델, 토크나이저 가져오기
  MODEL_NAME = "klue/bert-base"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # 데이터셋 만들기
  RE_train_dataset = load_and_tokenize_data("../dataset/train/train.csv", tokenizer)
  # RE_dev_dataset = load_and_tokenize_data("../dataset/train/dev.csv", tokenizer)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)
  
  # 하이퍼파라미터 지정
  model = load_model(MODEL_NAME, num_labels=30, device=device)
  print(model.config)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=20,              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True 
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_train_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train 함수 실행
  trainer.train()
  model.save_pretrained('./best_model')
def main():
  train()

if __name__ == '__main__':
  main()
