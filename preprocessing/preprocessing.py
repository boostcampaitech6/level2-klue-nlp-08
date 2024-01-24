import re

import pandas as pd

def clean_sentence(sentence):
    is_special_character = re.compile('[^@⊙\^#,~()\'\"/_;:*$?&%<>!.A-Za-z0-9ㄱ-ㅎ가-힣一-龥ぁ-んァ-ン\s]')
    sentence = is_special_character.sub(' ', sentence)
    
    return sentence

def preprocessing_dataset(dataset):
  '''
  처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜준다.

  Args:
      dataset (DataFrame): raw 데이터셋

  Returns:
      DataFrame: 'id', 'sentence', 'subject_entity', 'object_entity', 'label', ...를 반환한다.
  '''
  subject_entity = []
  subject_start_idx = []
  subject_end_idx = []
  subject_type = []

  object_entity = []
  object_start_idx = []
  object_end_idx = []
  object_type = []

  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    subject_dict, object_dict = eval(i), eval(j)

    subject_entity.append(subject_dict['word'])
    subject_start_idx.append(subject_dict['start_idx'])
    subject_end_idx.append(subject_dict['end_idx'])
    subject_type.append(subject_dict['type'])

    object_entity.append(object_dict['word'])
    object_start_idx.append(object_dict['start_idx'])
    object_end_idx.append(object_dict['end_idx'])
    object_type.append(object_dict['type'])

  out_dataset = pd.DataFrame({
                  'id':dataset['id'], 
                  'sentence':dataset['sentence'],
                  'subject_entity':subject_entity,
                  'subject_start_idx':subject_start_idx,
                  'subject_end_idx':subject_end_idx,
                  'subject_type':subject_type,
                  'object_entity':object_entity,
                  'object_start_idx':object_start_idx,
                  'object_end_idx':object_end_idx,
                  'object_type':object_type,
                  'label':dataset['label'],
                  })
  
  return out_dataset