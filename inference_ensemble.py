import ast
from collections import Counter
from typing import Dict, List

import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from data.dataset import RE_Dataset
from utils.utils import set_seed, num_to_label, load_config
from model.model import load_model
from preprocessing.tokenizer import TypedEntityMarkerPuncTokenizer

class EnsembleInference:
    def __init__(self, dataset_path, num_labels, token):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = dataset_path
        self.token = token
        self.num_labels = num_labels

    def get_models_and_tokenizers(self, model_dir:Dict):
        """
        vote_by_model 에서 호출되는 메소드.
        로드된 model 객체들의 리스트를 반환합니다.
        """
        output = []
        for dict in tqdm(model_dir.values(), desc="load_model", total=len(model_dir)):
            tokenizer = TypedEntityMarkerPuncTokenizer(dict['tokenizer'])
            model = load_model(dict['path'], self.num_labels, self.token)
            model.resize_token_embeddings(len(tokenizer.tokenizer))
            model.to(self.device)
            output.append([model, tokenizer])
        
        return output
    
    
    def vote_hard_model(self, average_probs:np.array, total_preds:np.array):
        """
        vote_by_model 에서 호출되는 메소드.
        total_preds 를 기반으로 hard voting 을 합니다.
        가장 높은 빈도의 label 이 여러개 있으면, average_prob 이 가장 높은 label 을 선택합니다.
        """
        final_pred = []
        
        for i, pred in enumerate(total_preds):
            label_count = Counter(pred)
            max_count = max(label_count.values())
            candidates = [label for label, count in label_count.items() if count == max_count]

            if len(candidates) > 1:
                highest_prob = -1
                selected_label = -1
                for candidate in candidates:
                    if average_probs[i][candidate] > highest_prob:
                        highest_prob = average_probs[i][candidate]
                        selected_label = candidate
                final_pred.append(selected_label)
            else:
                final_pred.append(candidates[0])
        
        return final_pred


    def vote_by_model(self, model_dir:List, voting_type:str, weights:List):
        """
        vote 에서 호출되는 메소드.
        주어진 모델 경로, 보팅 유형과 가중치를 기반으로 하드 보팅 혹은 소프트 보팅을 수행합니다.
        최종 확률과 라벨을 반환합니다.
        하드 보팅의 경우 라벨들의 평균 확률을 최종 확률로 사용합니다.
        """
        models = self.get_models_and_tokenizers(model_dir)
        
        total_preds = []
        total_probs = []
        for i, (model, tokenizer) in enumerate(models):
            tokenized_dataset = RE_Dataset(data_path=self.dataset_path, tokenizer=tokenizer, train=False)
            dataloader = DataLoader(tokenized_dataset, batch_size=64, shuffle=False)

            model_pred = []
            model_prob = []
            for _, data in enumerate(tqdm(dataloader, desc="voting progress")):
                model.eval()
                with torch.no_grad():
                    output = model(
                        input_ids = data['input_ids'].to(self.device),
                        attention_mask = data['attention_mask'].to(self.device),
                        token_type_ids=data['token_type_ids'].to(self.device)
                    )
                logits = output[0]
                weighted_prob = F.softmax(logits, dim=-1).detach().cpu().numpy() * weights[i]
                logits = logits.detach().cpu().numpy()
                preds = np.argmax(logits, axis=-1)

                model_pred.extend(preds)
                model_prob.append(weighted_prob)
            model_prob = np.concatenate(model_prob, axis=0)
            total_preds.append(model_pred)
            total_probs.append(model_prob)

        total_preds = np.array(total_preds).transpose()
        total_probs = np.array(total_probs)

        final_probs = np.mean(total_probs, axis = 0)

        if voting_type == 'hard':
            final_preds = self.vote_hard_model(final_probs, total_preds)
        else:
            final_preds = np.argmax(final_probs, axis=1)

        return final_probs, final_preds

    def vote_hard_csv(self, probs:pd.DataFrame):
        '''
        vote_by_csv 에서 호출되는 메소드.
        csv 의 확률 컬럼을 concat 한 DataFrame 을 바탕으로 하드 보팅을 수행합니다.
        '''
        num_of_columns = len(probs.columns)
        final_preds = []
        final_probs = []
        for _, row in tqdm(probs.iterrows(), desc="csv hard voting", total=len(probs)):
            votes = []
            prob_sum = np.zeros(len(ast.literal_eval(row[0])))
            for prob_str in row:
                prob_list = ast.literal_eval(prob_str)
                votes.append(np.argmax(prob_list))
                prob_sum += np.array(prob_list)
            average_prob = prob_sum / num_of_columns
            final_probs.append(average_prob)
            final_preds.append(Counter(votes).most_common(1)[0][0])
                 
        return final_probs, final_preds
    
    def vote_soft_csv(self, probs:pd.DataFrame, weight:List):
        '''
        vote_by_csv 에서 호출되는 메소드
        csv 의 확률 컬럼을 concat 한 DataFrame 과 가중치를 바탕으로 소프트 보팅을 수행합니다.
        '''
        final_probs = []
        final_preds = []
        for _, row in tqdm(probs.iterrows(), desc="csv soft voting", total=len(probs)):
            weighted_prob_sum = np.zeros(len(ast.literal_eval(row[0])))
            for i, prob_str in enumerate(row):
                prob_list = ast.literal_eval(prob_str)
                weighted_prob_sum += np.array(prob_list) * weight[i]
            weighted_probs = weighted_prob_sum / sum(weight)
            final_probs.append(weighted_probs)
            final_preds.append(np.argmax(weighted_probs))
        
        return final_probs, final_preds

    def vote_by_csv(self, csv_path:List, vote_type:str, weight:List):
        '''
        vote 에서 호출되는 메소드
        주어진 csv_path 로 데이터들을 불러와 probs 컬럼을 concat 하고, 보팅 유형에 따라 적절한 메소드에 전달합니다.
        '''
        prob_columns = pd.DataFrame()
        for path in csv_path.values():
            df = pd.read_csv(path)
            prob_column = df['probs']
            prob_columns = pd.concat([prob_columns, prob_column], axis=1)

        if vote_type == 'hard':
            final_probs, final_preds = self.vote_hard_csv(prob_columns)
        else:
            final_probs, final_preds = self.vote_soft_csv(prob_columns, weight)

        return final_probs, final_preds

    def vote(self, mode:str, voting_type:str, path:dict, w=None):
        '''
        주어진 mode 에 따라 각 모드에 맞는 메소드를 호출하고 voting 결과를 반환 받습니다.
        voting 결과를 바탕으로 id, pred_label, probs 컬럼으로 구성된 데이터 프레임을 최종 반환합니다.
        '''
        modes = ['csv', 'model']
        types = ['hard', 'soft']
        assert mode in modes, print("Warning: Invalid mode")
        assert voting_type in types, print("Warning: Invalid type")

        weight = w
        if weight is not None:
            assert len(weight) == len(path[mode].keys()), print(f"Warning: The number of elements in 'weight' must be equal to the number of '{mode}'.")
            assert voting_type =="soft", print("Warning: Hard voting can't use weight")
        else:
            weight = [1.0] * len(path[mode].keys())

        if mode == 'model':
            probs, indices = self.vote_by_model(path['model'], voting_type, weight)
        else:
            probs, indices = self.vote_by_csv(path['csv'], voting_type, weight)

        labels = num_to_label(indices)
        test_id = pd.read_csv(self.dataset_path)['id'].values

        probs_str = ["["+', '.join(map(str, prob)) +"]" for prob in probs]

        answer = pd.DataFrame({'id': test_id, 'pred_label': labels, 'probs': probs_str})

        return answer

                

if __name__ == '__main__':
    set_seed(42)
    
    CONFIG_PATH = './training_recipes/inference_config.yaml'
    config = load_config(CONFIG_PATH, 'inference_ensemble_config')

    ensemble = EnsembleInference(config['test_dataset_path'], config['num_labels'], config['token'])
    output = ensemble.vote(
        mode=config['mode'], # mode: csv, model
        voting_type=config['voting_type'], # type: hard, soft
        path=config['path'], # path: config['model_dir'], config['csv_path']
        w=config['weight'] # w(weight): hard type 을 사용하면 weight 를 null 로 설정해주세요.
    )

    output.to_csv(config['output_path'], index=False)
