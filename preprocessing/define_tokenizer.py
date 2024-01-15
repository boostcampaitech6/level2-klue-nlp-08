from tokenizer import TypedEntityMarkerPuncTokenizer, TypedEntityMarkerTokenizer, ConcatEntityTokenizer
def load_tokenizer(tokenizer_type, tokenizer_name, add_query):
    if tokenizer_type == 'TypedEntityMarkerPuncTokenizer':
        return TypedEntityMarkerPuncTokenizer(tokenizer_name, add_query)
    if tokenizer_type == 'TypedEntityMarkerTokenizer':
        return TypedEntityMarkerTokenizer(tokenizer_name)
    if tokenizer_type == 'ConcatEntityTokenizer':
        return ConcatEntityTokenizer(tokenizer_name)