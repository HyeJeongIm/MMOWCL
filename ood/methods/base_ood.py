import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

class BaseOODDetector(ABC):
    """Base class for OOD detection methods"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.name = self.__class__.__name__
    
    @abstractmethod
    def compute_scores(self, loader, **kwargs):
        """Compute OOD scores for given data loader"""
        pass
    
    def get_predictions_and_features(self, loader):
        """Extract predictions and features from model"""
        self.model.eval()
        predictions = []
        features = []
        logits = []
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                # Handle multimodal inputs
                if isinstance(inputs, dict):
                    for m in inputs:
                        inputs[m] = inputs[m].to(self.device)
                else:
                    inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                logits.append(outputs["logits"].cpu())
                
                if "features" in outputs:
                    features.append(outputs["features"].cpu())
                
                pred = F.softmax(outputs["logits"], dim=1)
                predictions.append(pred.cpu())
        
        predictions = torch.cat(predictions, dim=0)
        logits = torch.cat(logits, dim=0)
        
        if features:
            features = torch.cat(features, dim=0)
            return predictions, features, logits
        else:
            return predictions, None, logits
        
    def compute_scores_batch(self, inputs):
        """Compute OOD scores for a single batch"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs["logits"]
            return self._compute_scores_from_logits(logits)

    @abstractmethod
    def _compute_scores_from_logits(self, logits):
        """Compute scores from logits - to be implemented by subclasses"""
        pass