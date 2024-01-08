# dataset, tokenizer를 받아 dataset의 각 문장을 토큰화
def tokenized_dataset(dataset, tokenizer):
  '''
  tokenizer에 따라 sentence를 tokenizing 진행한다.

  Args:
      dataset (dataframe): 전처리된 데이터셋을 dataframe 형태로 불러온다.
      tokenizer (_type_): 모델에 해당하는 tokenizer를 불러온다.

  Returns:
      tensor: 'subject_entity', [SEP], 'object_entity', 'sentence'를 토큰화화여 반환한다.
  '''
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    # temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=128,
      add_special_tokens=True,
      )
  return tokenized_sentences