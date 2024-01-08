    
# def tokenized_dataset(dataset, tokenizer):
#     concat_entity = []
#     for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
#         temp = e01 + '[SEP]' + e02
#         concat_entity.append(temp)

#     tokenized_sentences = tokenizer(
#         concat_entity,
#         list(dataset['sentence']),
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=128,
#         add_special_tokens=True,
#         )
    
#     return tokenized_sentences


def tokenized_dataset_concat_entity(dataset, tokenizer):
    '''
    Args:
        dataset : dataframe 
        tokenizer : model tokenizer

    Returns:
        tensor: [CLS] 'subject_entity', [SEP], 'object_entity', 'sentence'를 토큰화화여 반환한다.
    '''
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,)
    
    return tokenized_sentences
