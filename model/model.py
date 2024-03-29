from transformers import AutoConfig, AutoModelForSequenceClassification

def load_model(model_name, num_labels, token=''):
    """
    Hugging Face 트랜스포머 라이브러리에서 사전 학습된 시퀀스 분류 모델을 로드합니다.
    
    Args:
        model_name (str): 불러올 사전 학습된 모델의 이름 또는 경로입니다.
        num_labels (int): 시퀀스 분류 작업의 라벨 수입니다.
    
    Returns:
        transformers.modeling_auto.AutoModelForSequenceClassification: 지정된 수의 라벨에 대해 구성된 사전 학습된 시퀀스 분류 모델입니다.
    """
    
    args = {'pretrained_model_name_or_path': model_name}
    if '2024-level2-re-nlp-8' in model_name:
        args['token'] = token

    model_config = AutoConfig.from_pretrained(**args)
    model_config.num_labels = num_labels

    args['config'] = model_config

    model = AutoModelForSequenceClassification.from_pretrained(**args)
    
    return model
