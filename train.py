import torch
import wandb
from transformers import Trainer, TrainingArguments

from data.dataset import RE_Dataset
from utils.utils import set_seed
from model.model import load_model
from preprocessing.tokenizer import TypedEntityMarkerPuncTokenizer
from model.model_type import train_model
from utils.utils import load_config
    
def train():
    CONFIG_PATH = './config.yaml'
    config = load_config(CONFIG_PATH, 'train_config')

    wandb.init(
        project="RE",
        name=config['model_save_name'],
        entity="goghinarles"
        )
    
    set_seed(42)        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('start training on :',device)


    tokenizer = TypedEntityMarkerPuncTokenizer(config['model_name'], add_query=config['add_query'])
    # Prepare dataset
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
    training_args = TrainingArguments(**load_config(CONFIG_PATH, 'train_args'), report_to='wandb')

    trainer = train_model(m_type = 'focal_loss',    # 'none', 'none-ealry_stopping', 'focal_loss', 'focal_loss-early_stopping'
                          model=model, 
                          training_args=training_args, 
                          train_dataset=RE_train_dataset, 
                          eval_dataset= RE_valid_dataset).model_type()
    # train start
    trainer.train()
    model.save_pretrained(f"./best_model/{config['model_save_name']}")
    
if __name__ == '__main__':
    train()