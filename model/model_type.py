from transformers import Trainer
from utils.metrics import compute_metrics
import torch.nn.functional as F
import torch

# focal loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # loss = FocalLoss(outputs[0], labels)
        log_prob = F.log_softmax(outputs[0], dim=-1)
        prob = torch.exp(log_prob)
        gamma = 2
        loss = F.nll_loss(
            ((1 - prob) ** gamma) * log_prob,
            labels)
        return (loss, outputs) if return_outputs else loss
    
# Model Trainer
class train_model():
    def __init__(self, m_type, model, training_args, train_dataset, eval_dataset):
        self.m_type = m_type
        self.model = model
        self.training_args = training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics

    def model_type(self):
        if self.m_type == 'none':
            return Trainer(    # custom trainer 안쓰고 싶으면 CustomTrainer -> Trainer 로 변경
                model=self.model,                         # the instantiated 🤗 Transformers model to be trained
                args=self.training_args,                  # training arguments, defined above
                train_dataset=self.train_dataset,         # training dataset
                eval_dataset=self.eval_dataset,             # evaluation dataset
                compute_metrics=compute_metrics         # define metrics function
            )
        if self.m_type == 'focal_loss':
            return CustomTrainer(    # custom trainer 안쓰고 싶으면 CustomTrainer -> Trainer 로 변경
                model=self.model,                         # the instantiated 🤗 Transformers model to be trained
                args=self.training_args,                  # training arguments, defined above
                train_dataset=self.train_dataset,         # training dataset
                eval_dataset=self.eval_dataset,             # evaluation dataset
                compute_metrics=compute_metrics         # define metrics function
            )