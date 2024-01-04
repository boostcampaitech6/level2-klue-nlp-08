from transformers import AutoConfig, AutoModelForSequenceClassification

def load_model(model_name, num_labels, device):
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = num_labels

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
    model.to(device)

    return model