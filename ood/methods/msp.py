import torch
import torch.nn.functional as F
import numpy as np
from .base_ood import BaseOODDetector

class MSPDetector(BaseOODDetector):
    """Maximum Softmax Probability (MSP) Detector"""
    
    def compute_scores(self, loader, **kwargs):
        """Compute MSP scores"""
        self.model.eval()
        msp_scores = []
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                # Handle multimodal inputs
                if isinstance(inputs, dict):
                    for m in inputs:
                        inputs[m] = inputs[m].to(self.device)
                else:
                    inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                logits = outputs["logits"]
                
                # Compute MSP scores
                probs = F.softmax(logits, dim=1)
                max_probs = probs.max(1)[0]
                msp_scores.append(max_probs.cpu().numpy())
        
        return np.concatenate(msp_scores)
    
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