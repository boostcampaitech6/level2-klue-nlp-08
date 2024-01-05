import pickle
from load_data import load_data, tokenized_dataset, RE_Dataset

# dict_label_to_num 사전을 이용해 문자 라벨을 숫자 인덱스로 변환
def label_to_num(label):
    """
    미리 정의된 mapping dict를 사용하여 라벨 목록을 해당 인덱스 값으로 변환합니다.
    
    args:
        label: 변환할 라벨 목록입니다.
    
    returns:
        num_label: 라벨 목록에 해당하는 변환된 인덱스 값의 목록입니다.
    """
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
    
    return num_label

def load_and_tokenize_data(file_path, tokenizer):
    """
    지정된 토크나이저를 사용하여 파일에서 데이터를 로드하고 토큰화합니다.
    
    args:
        file_path: 데이터셋이 포함된 파일의 경로입니다.
        tokenizer: 토큰화에 사용할 토크나이저입니다.
      
    returns:
        RE_Dataset: 각각의 인덱스화된 라벨과 토큰화된 데이터가 합해진 데이터 세트입니다.
    """
    dataset = load_data(file_path)
    labels = label_to_num(dataset['label'].values)
    tokenized_data = tokenized_dataset(dataset, tokenizer)
    return RE_Dataset(tokenized_data, labels)