import pickle

# dict_label_to_num 사전을 이용해 문자 라벨을 숫자 인덱스로 변환
def label_to_num(label):
    """
    미리 정의된 mapping dict를 사용하여 라벨 목록을 해당 인덱스 값으로 변환합니다.
    
    Args:
        label (list): 변환할 라벨 목록입니다.
    
    Returns:
        list: 라벨 목록에 해당하는 변환된 인덱스 값의 목록입니다.
    """
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
    
    return num_label

# def num_to_label(label):