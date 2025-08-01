import torch
import torch.nn.functional as F
import numpy as np
from .base_ood import BaseOODDetector

class EnergyDetector(BaseOODDetector):
    """Energy-based Out-of-Distribution Detection"""
    
    def __init__(self, model, device='cuda', temperature=1.0):
        super().__init__(model, device)
        self.temperature = temperature
    
    def compute_scores(self, loader, **kwargs):
        """Compute Energy scores"""
        self.model.eval()
        energy_scores = []
        
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
                
                # Compute energy score: -T * log(sum(exp(logits/T)))
                energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
                energy_scores.append(energy.cpu().numpy())
        
        return np.concatenate(energy_scores)
    
    def _compute_scores_from_logits(self, logits):
        """Compute Energy scores from logits"""
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        return energy.cpu().numpy()

    def compute_scores_batch(self, inputs):
        """Compute Energy scores for a single batch"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs["logits"]
            return self._compute_scores_from_logits(logits)