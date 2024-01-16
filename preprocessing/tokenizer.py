from tqdm import tqdm
from transformers import AutoTokenizer

def load_tokenizer(tokenizer_type, tokenizer_name, add_query=False):
    if tokenizer_type == 'TypedEntityMarkerPuncTokenizer':
        return TypedEntityMarkerPuncTokenizer(tokenizer_name, add_query)
    if tokenizer_type == 'TypedEntityMarkerTokenizer':
        return TypedEntityMarkerTokenizer(tokenizer_name)
    if tokenizer_type == 'ConcatEntityTokenizer':
        return ConcatEntityTokenizer(tokenizer_name)

class TypedEntityMarkerPuncTokenizer():
    def __init__(self, tokenizer_name, add_query=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.add_query = add_query
        self.type2word = {"ORG": "단체", "PER": "사람", "DAT": "날짜", 
                          "LOC": "위치", "POH": "기타", "NOH": "수량"}
        
    def get_query(self, data_series) -> str:
        '''
        return "@ ⊙ 사람 ⊙ 이순신 @ 과 # ^ 시대 ^ 조선 # 의 관계는 무엇인가?"
        '''

        sentence = data_series['sentence']
        o_start = int(data_series['object_start_idx'])
        o_end = int(data_series['object_end_idx'])
        o_type = data_series['object_type']
        s_start = int(data_series['subject_start_idx'])
        s_end = int(data_series['subject_end_idx'])
        s_type = data_series['subject_type']

        if o_start < s_start:
            object = sentence[o_start:o_end+1]
            subject = sentence[s_start:s_end+1]

            return f"@ ⊙ {self.type2word[o_type]} ⊙ " + object + " @ 과 " + \
                    f"# ^ {self.type2word[s_type]} ^ " + subject + f" # 사이의 관계는 무엇인가?"
        else:
            subject = sentence[s_start:s_end+1]
            object = sentence[o_start:o_end+1]

            return f"# ^ {self.type2word[s_type]} ^ " + subject + " # 과 " + \
                    f"@ ⊙ {self.type2word[o_type]} ⊙ " + object + f" @ 사이의 관계는 무엇인가?"


    def mark_entities(self, data_series) -> str:
        ''' 
            @ ⊙ 사람 ⊙ 이순신 @ 장군은 # ^ 시대 ^ 조선 # 출신 이다 
        '''

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
            
            return s1 + f"@ ⊙ {self.type2word[o_type]} ⊙ " + s2 + " @ " + \
                s3 + f"# ^ {self.type2word[s_type]} ^ " + s4 + f" # " + s5
            
        else:
            s1 = sentence[:s_start]
            s2 = sentence[s_start:s_end+1]
            s3 = sentence[s_end+1 :o_start]
            s4 = sentence[o_start:o_end+1]
            s5 = sentence[o_end+1:]
            
            return s1 + f"# ^ {self.type2word[s_type]} ^ " + s2 + " # " + \
                s3 + f"@ ⊙ {self.type2word[o_type]} ⊙ " + s4 + f" @ " + s5
                    
    def tokenize(self, dataset):

        marked_sentence = []
        for _, data in tqdm(dataset.iterrows(), desc="adding typed entity marker", total=len(dataset)):
            answer = self.mark_entities(data)
            if self.add_query:
                marked_sentence.append([self.get_query(data), self.mark_entities(data)])
            else:
                marked_sentence.append(self.mark_entities(data))

        tokenized_sentences = self.tokenizer(
            marked_sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
            )
        
        return tokenized_sentences

class TypedEntityMarkerTokenizer():
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.special_tokens_dict = {'additional_special_tokens': ['[/S:LOC]', '[S:LOC]',
                                    '[S:PER]', '[S:ORG]', '[/S:PER]', '[/S:ORG]', 
                                    '[/O:POH]', '[/O:LOC]','[/O:ORG]', '[O:POH]', 
                                    '[/O:DAT]', '[/O:PER]','[O:ORG]', '[O:NOH]', 
                                    '[/O:NOH]', '[O:PER]', '[O:LOC]', '[O:DAT]']}
        self.tokenizer.add_special_tokens(self.special_tokens_dict)

    def mark_entities(self, data_series) -> str:
        ''' [S:PERSON] 이순신 [/S:PERSON] 장군은 [O:OCUP] 조선 [/O:OCUP] 출신 이다 '''

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
        
    def tokenize(self, dataset):
        marked_sentence = []
        for _, data in tqdm(dataset.iterrows(), desc="adding typed entity marker", total=len(dataset)):
            marked_sentence.append(self.mark_entities(data))
            
        tokenized_sentences = self.tokenizer(
            marked_sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
            )

        return tokenized_sentences

class ConcatEntityTokenizer():
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(self, dataset):
        ''' tensor: [CLS] 이순신 [SEP] 무신 [SEP] 이순신 장군은 고려시대 무신이다. '''
        
        concat_entity = []
        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
            temp = e01 + '[SEP]' + e02
            concat_entity.append(temp)

        tokenized_sentences = self.tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,)
        
        return tokenized_sentences