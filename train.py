import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
import numpy as np
import random
from load_data import *
from label_utils import *
from metrics import compute_metrics
from model import load_model

from tokenizing import *
from preprocessing import *

# 시드 설정
def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train():
    """
    KLUE 데이터 세트를 사용하여 관계 추출 모델을 훈련합니다.

    이 함수는 무작위 시드를 설정하고, 사전 학습된 모델과 토크나이저를 로드하고, 학습 데이터 세트를 생성합니다.
    하이퍼파라미터와 훈련 인자를 지정한 다음, 트랜스포머 라이브러리의 Trainer를 사용하여 모델을 훈련합니다.

    훈련된 모델은 './best_model' 디렉터리에 저장됩니다.
    
    참고 자료:
        - 허깅 페이스 트랜스포머 라이브러리: https://huggingface.co/transformers/
        - KLUE 관계 추출 데이터 세트: https://klue-benchmark.com/tasks/67/overview
        - 허깅 페이스 트랜스포머 training options: https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
        
    Raises:
        ImportError: transformers, torch, numpy, 또는 sklearn이 설치되지 않은 경우 발생합니다.
    """
    # 시드 설정
    set_seed(42)
    # 모델, 토크나이저 가져오기
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = load_data("../dataset/train/train.csv")
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset['label'].values)
    # dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    # 디바이스에 올리기
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # 모델 불러오기
    model = load_model(MODEL_NAME, num_labels=30)
    model.to(device)
    print(model.config)
    
    # 훈련 인자 지정
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
    # Trainer 선언
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
