from tqdm import tqdm

def mark_entities(data_series) -> str:
    """
    data_series 에서 entity 의 시작 인덱스, 끝 인덱스와 타입을 추출하고,
    추출한 정보를 바탕으로 typed entity marker 가 추가된 문장을 반환한다.
    """
    sentence = data_series['sentence']
    o_start = int(data_series['object_start_idx'])
    o_end = int(data_series['object_end_idx'])
    o_type = data_series['object_type']
    s_start = int(data_series['subject_start_idx'])
    s_end = int(data_series['subject_end_idx'])
    s_type = data_series['subject_type']

    if o_start < s_start:
        s1 = sentence[:o_start]
        s2 = sentence[o_start:o_end+1]
        s3 = sentence[o_end+1:s_start]
        s4 = sentence[s_start:s_end+1]
        s5 = sentence[s_end+1:]

        return s1 + f"[O:{o_type}] " + s2 + f" [/O:{o_type}] " + \
               s3 + f"[S:{s_type}] " + s4 + f" [/S:{s_type}] " + s5
    else:
        s1 = sentence[:s_start]
        s2 = sentence[s_start:s_end+1]
        s3 = sentence[s_end+1 :o_start]
        s4 = sentence[o_start:o_end+1]
        s5 = sentence[o_end+1:]

        return s1 + f"[S:{s_type}] " + s2 + f" [/S:{s_type}] " + \
               s3 + f"[O:{o_type}] " + s4 + f" [/O:{o_type}] " + s5
    
def tokenized_dataset_type_entity_marker(dataset, tokenizer):
    marked_sentence = []
    for _, data in tqdm(dataset.iterrows(), desc="adding typed entity marker", total=len(dataset)):
        special_tokens_dict = {'additional_special_tokens': [f"[O:{data['object_type']}]",
                                                            f"[/O:{data['object_type']}]",
                                                            f"[S:{data['subject_type']}]",
                                                            f"[/S:{data['subject_type']}]"]}
        num = tokenizer.add_special_tokens(special_tokens_dict, False)
        marked_sentence.append(mark_entities(data))
        
    tokenized_sentences = tokenizer(
        marked_sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
        )
    
    print(tokenized_sentences['input_ids'][0])
    print(tokenizer.decode(tokenized_sentences['input_ids'][0]))

    return tokenized_sentences

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
