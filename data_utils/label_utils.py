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

def num_to_label(label):
  '''
  dict를 이용해 숫자로 된 class -> 문자열 label로 반환

  Args:
      label (_type_): _description_

  Returns:
      List: 문자열로 되어있는 label list 반환
  '''

  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label