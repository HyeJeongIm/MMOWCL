import torch
import torch.nn.functional as F
import numpy as np
from .base_ood import BaseOODDetector

class ODINDetector(BaseOODDetector):
    """ODIN: Out-of-Distribution Detector for Neural Networks (Simplified version)"""
    
    def __init__(self, model, device='cuda', temperature=1000, epsilon=0.0014):
        super().__init__(model, device)
        self.temperature = temperature
        self.epsilon = epsilon
    
    def compute_scores(self, loader, **kwargs):
        """Compute ODIN scores (simplified version without gradient perturbation)"""
        self.model.eval()
        odin_scores = []
        
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
                # Temperature scaling only (without gradient perturbation for now)
                logits = logits / self.temperature
                
                # Compute ODIN score (max softmax probability)
                scores = F.softmax(logits, dim=1).max(1)[0]
                odin_scores.append(scores.cpu().numpy())
        
        return np.concatenate(odin_scores)
    
    def _compute_scores_from_logits(self, logits):
        """Compute ODIN scores from logits"""
        logits = logits / self.temperature
        scores = F.softmax(logits, dim=1).max(1)[0]
        return scores.cpu().numpy()

    def compute_scores_batch(self, inputs):
        """Compute ODIN scores for a single batch"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs["logits"]
            return self._compute_scores_from_logits(logits)