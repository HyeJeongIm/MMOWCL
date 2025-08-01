import torch
import torch.nn.functional as F
import numpy as np
from .base_ood import BaseOODDetector

class MSPDetector(BaseOODDetector):
    """Maximum Softmax Probability (MSP) Detector"""
    
    def compute_scores(self, loader, **kwargs):
        """Compute MSP scores"""
        predictions, _, _ = self.get_predictions_and_features(loader)
        return predictions.max(1)[0].numpy()
    
    def _compute_scores_from_logits(self, logits):
        """Compute MSP scores from logits"""
        probs = F.softmax(logits, dim=1)
        scores = probs.max(1)[0]
        return scores.cpu().numpy()

    def compute_scores_batch(self, inputs):
        """Compute MSP scores for a single batch"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs["logits"]
            return self._compute_scores_from_logits(logits)