from transformers import Trainer, EarlyStoppingCallback
from utils.metrics import compute_metrics
import torch.nn.functional as F
import torch

# focal loss
class FocalLoss(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
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
            return Trainer(    
                model=self.model,                         # the instantiated 🤗 Transformers model to be trained
                args=self.training_args,                  # training arguments, defined above
                train_dataset=self.train_dataset,         # training dataset
                eval_dataset=self.eval_dataset,             # evaluation dataset
                compute_metrics=compute_metrics         # define metrics function
            )
        if self.m_type == 'none-early_stopping':
            return Trainer(    
                model=self.model,                         # the instantiated 🤗 Transformers model to be trained
                args=self.training_args,                  # training arguments, defined above
                train_dataset=self.train_dataset,         # training dataset
                eval_dataset=self.eval_dataset,             # evaluation dataset
                compute_metrics=compute_metrics         # define metrics function
            )
        if self.m_type == 'focal_loss':
            return FocalLoss(    
                model=self.model,                         # the instantiated 🤗 Transformers model to be trained
                args=self.training_args,                  # training arguments, defined above
                train_dataset=self.train_dataset,         # training dataset
                eval_dataset=self.eval_dataset,             # evaluation dataset
                compute_metrics=compute_metrics         # define metrics function
            )
        if self.m_type == 'focal_loss-early_stopping':
            return FocalLoss(    
                model=self.model,                         # the instantiated 🤗 Transformers model to be trained
                args=self.training_args,                  # training arguments, defined above
                train_dataset=self.train_dataset,         # training dataset
                eval_dataset=self.eval_dataset,             # evaluation dataset
                compute_metrics=compute_metrics,         # define metrics function,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )