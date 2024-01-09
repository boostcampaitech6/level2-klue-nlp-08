import torch
import pickle as pickle
from transformers import Trainer, TrainingArguments

from data.dataset import RE_Dataset
from utils.utils import set_seed
from model.model import load_model
from utils.metrics import compute_metrics
from preprocessing.tokenizer import TypedEntityMarkerTokenizer, TypedEntityMarkerPuncTokenizer

def train():
    set_seed(42)        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('start training on :',device)
    MODEL_NAME = "klue/roberta-large"

    tokenizer = TypedEntityMarkerPuncTokenizer(MODEL_NAME)
    # Prepare dataset
    RE_train_dataset = RE_Dataset(data_path="./dataset/train/train_split_v1.csv", 
                                  tokenizer=tokenizer)
    RE_valid_dataset = RE_Dataset(data_path="./dataset/valid/valid_split_v1.csv", 
                                  tokenizer=tokenizer)
    
    # Load Model
    model = load_model(MODEL_NAME, num_labels=30)
    model.to(device)
    model.resize_token_embeddings(len(tokenizer.tokenizer))
    print(model.config)
    
    # TrainingArguments setup
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=20,              # total number of training epochs
        learning_rate=2e-5,               # learning_rate
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.05,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 500,            # evaluation step.
        load_best_model_at_end = True 
    )

    # Model Trainer
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_valid_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # train start
    trainer.train()
    model.save_pretrained('./best_model')
    
if __name__ == '__main__':
    train()
