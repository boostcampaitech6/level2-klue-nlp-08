import torch
import wandb
import pickle as pickle
from transformers import Trainer, TrainingArguments

from utils.utils import set_seed, load_config
from model.model import load_model
from utils.metrics import compute_metrics
from data.dataset import RE_Dataset
from trainer.trainer import FocalLossTrainer
from preprocessing.define_tokenizer import load_tokenizer

def train():
    set_seed(42)        
    
    CONFIG_PATH = './training_recipes/train_config.yaml'
    config = load_config(CONFIG_PATH, 'train_config')

    # wandb project name
    wandb.init(project="KLUE-RE-3") 
    wandb.run.name = config['model_name']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('start training on :',device)

    # tokenizer = TypedEntityMarkerPuncTokenizer(config['model_name'], add_query=config['add_query'])
    tokenizer = load_tokenizer(config['tokenizer_type'], config['model_name'], add_query=config['add_query'])

    RE_train_dataset = RE_Dataset(config['train_dataset_path'], 
                                  tokenizer=tokenizer)
    RE_valid_dataset = RE_Dataset(config['valid_dataset_path'], 
                                  tokenizer=tokenizer)
    
    # Load Model
    model = load_model(config['model_name'], num_labels=30)
    model.to(device)
    model.resize_token_embeddings(len(tokenizer.tokenizer))
    print(model.config)

    # TrainingArguments setup
    training_args = TrainingArguments(
        **load_config(CONFIG_PATH, 'train_args'), 
        report_to='wandb'
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_valid_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # train start
    trainer.train()
    model.save_pretrained(f"./best_model/{config['model_save_name']}")
    
if __name__ == '__main__':
    train()
