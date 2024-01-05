import pandas as pd

def preprocessing_dataset(dataset):
  '''
  처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜준다.

  Args:
      dataset (DataFrame): raw 데이터셋

  Returns:
      DataFrame: 'id', 'sentence', 'subject_entity', 'object_entity', 'label'를 반환한다.
  '''
  subject_entity = []
  object_entity = []

  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset