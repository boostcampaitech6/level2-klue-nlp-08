import torch
import torch.nn.functional as F
from transformers import Trainer

class FocalLossTrainer(Trainer):
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
    