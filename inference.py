import torch
import pandas as pd
import torch.nn.functional as F
import pickle as pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader

from data.dataset import RE_Dataset
from utils.utils import set_seed, num_to_label

def inference(model, tokenized_sent, device):
    '''
    Args:
        model : AutoModelForSequenceClassification
        tokenized_sent: RE_Dataset
        device: cuda or cpu

    Returns:
        List: prediction label
        List: prediction probabilities
    '''
    dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False)
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

if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    MODEL_NAME = "klue/roberta-base"
    # model_dir = './results/roberta-base1-focal/checkpoint-9000/'
    model_dir = './best_model/'
    test_dataset_dir = "./dataset/test/test_data.csv"
    output_path = f'./prediction/submission-focal-best.csv'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # special_tokens_dict = {'additional_special_tokens': ['[LOC]', '[PER]', '[ORG]', '[POH]', '[DAT]', '[NOH]', '∴','∵']}
    special_tokens_dict = {'additional_special_tokens': ['[/S:LOC]', '[S:LOC]', '[S:PER]', '[S:ORG]', '[/S:PER]', 
                                                         '[/S:ORG]', '[/O:POH]', '[/O:LOC]', '[/O:ORG]', '[O:POH]', 
                                                         '[/O:DAT]', '[/O:PER]', '[O:ORG]', '[O:NOH]', '[/O:NOH]', 
                                                         '[O:PER]', '[O:LOC]', '[O:DAT]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    ## load test datset
    Re_test_dataset = RE_Dataset(test_dataset_dir ,tokenizer, train=False)
    test_id, _, _ = Re_test_dataset.get_data_and_label()
    
    ## predict answer
    pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
    pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
    
    ## make csv file with predicted answer
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
    output.to_csv(output_path, index=False)
    