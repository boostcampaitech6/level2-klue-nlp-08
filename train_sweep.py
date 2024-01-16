import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
import numpy as np
import random
from data.dataset import RE_Dataset
from utils.metrics import compute_metrics
from model.model import load_model
from preprocessing.tokenizer import TypedEntityMarkerPuncTokenizer

import wandb

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
    # wandb init
    wandb.init()

    # 시드 설정
    set_seed(42)
    # 모델, 토크나이저 가져오기
    MODEL_NAME = "klue/bert-base"
    tokenizer = TypedEntityMarkerPuncTokenizer(MODEL_NAME)

    # load dataset
    RE_train_dataset = RE_Dataset(data_path="./dataset/train/train_split_v1.csv", 
                                  tokenizer=tokenizer)
    RE_valid_dataset = RE_Dataset(data_path="./dataset/valid/valid_split_v1.csv", 
                                  tokenizer=tokenizer)

    # 디바이스에 올리기
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    
    # 모델 불러오기
    model = load_model(MODEL_NAME, num_labels=30)
    model.to(device)
    model.resize_token_embeddings(len(tokenizer.tokenizer))
    # print(model.config)
    
    # 훈련 인자 지정
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=40,                 # model saving step.
        num_train_epochs=3,              # total number of training epochs
        learning_rate=wandb.config.lr,               # learning_rate
        per_device_train_batch_size=wandb.config.batch_size,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=5,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=20,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 20,            # evaluation step.
        load_best_model_at_end = True,
        report_to='wandb',
    )
    # Trainer 선언
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_valid_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # train 함수 실행
    trainer.train()
    model.save_pretrained('./best_model')
    
def main():
    # wandb sweep할 하이퍼파라미터 설정
    sweep_config = {    # yaml 파일에서 불러오기
        'method': 'random',
        'metric': {
            'name':'eval/micro f1 score', 
            'goal':'maximize'
        },
        'parameters': {
            'lr':{
                'distribution': 'uniform',  
                'min':5e-5,                 
                'max':5e-2                 
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            # 'warmup_steps': {
            #     'values': [0, 100, 500]
            # }, 
            # 'epochs': {
            #     'values': [5, 10, 20]
            # }
        }        
    }
    # sweep_id 생성
    sweep_id = wandb.sweep(
        project='KLUE-RE',
        sweep=sweep_config,  
    )

    # sweep agent 생성
    wandb.agent(
        sweep_id=sweep_id,      
        function=train,   
        count=3 # yaml 파일에서 불러오기          
    )

if __name__ == '__main__':
    main()
