import torch
import pickle as pickle
from transformers import Trainer, TrainingArguments

from data.dataset import RE_Dataset
from utils.utils import set_seed
from model.model import load_model
from preprocessing.tokenizer import TypedEntityMarkerTokenizer, TypedEntityMarkerPuncTokenizer
from model.model_type import train_model
    
def train():
    set_seed(42)        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('start training on :',device)
    MODEL_NAME = "klue/roberta-base"

    tokenizer = TypedEntityMarkerPuncTokenizer(MODEL_NAME)
    # Prepare dataset
    RE_train_dataset = RE_Dataset(data_path="./dataset/train/train_split_v1.csv", 
                                  tokenizer=tokenizer)
    RE_valid_dataset = RE_Dataset(data_path="./dataset/train/valid_split_v1.csv", 
                                  tokenizer=tokenizer)
    
    # Load Model
    model = load_model(MODEL_NAME, num_labels=30)
    model.to(device)
    model.resize_token_embeddings(len(tokenizer.tokenizer))
    print(model.config)
    
    # TrainingArguments setup
    training_args = TrainingArguments(
        output_dir='./results/roberta-base-focal-128-0.01-2e-5',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=30,              # total number of training epochs
        learning_rate=2e-5,               # learning_rate
        per_device_train_batch_size=128,  # batch size per device during training
        per_device_eval_batch_size=128,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 500,            # evaluation step.
        load_best_model_at_end = True,
        metric_for_best_model='micro f1 score'
    )

    trainer = train_model(m_type = 'focal_loss',    # 'none', 'none-ealry_stopping', 'focal_loss', 'focal_loss-early_stopping'
                          model=model, 
                          training_args=training_args, 
                          train_dataset=RE_train_dataset, 
                          eval_dataset= RE_valid_dataset).model_type()
    # train start
    trainer.train()
    model.save_pretrained('./best_model/roberta-base-focal-128-0.01-2e-5')
    
if __name__ == '__main__':
    train()