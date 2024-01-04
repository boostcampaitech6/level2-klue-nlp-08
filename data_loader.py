import pickle
from load_data import load_data, tokenized_dataset, RE_Dataset

# dict_label_to_num 사전을 이용해 문자 라벨을 숫자 인덱스로 변환
def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def load_and_tokenize_data(file_path, tokenizer):
    dataset = load_data(file_path)
    labels = label_to_num(dataset['label'].values)
    tokenized_data = tokenized_dataset(dataset, tokenizer)
    return RE_Dataset(tokenized_data, labels)