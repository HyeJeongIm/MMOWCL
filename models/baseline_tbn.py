# models/baseline.py

import torch
import torch.nn as nn
import copy

from models.backbones import get_backbone
from models.fusion import get_fusion
from models.classifier.classification_tbn import ClassificationTBN


class BaselineTBN(nn.Module):
    """Multi-modal baseline network with backbone, fusion, and classifier"""
    
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_segments = args["num_segments"]
        self.modality = args["modality"]
        self.backbone_name = args["backbone"]  # e.g., 'tbn'
        self.midfusion = args["midfusion"]     # e.g., 'concat'
        self.dropout = args["dropout"]
        self.consensus_type = args["consensus_type"]
        self.before_softmax = args["before_softmax"]

        if not self.before_softmax and self.consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        # Initialize backbone network for feature extraction
        self.backbone = get_backbone(args)  # output: feature list per modality

        # Initialize fusion network to combine multi-modal features
        self.fusion = get_fusion(
            midfusion=self.midfusion,
            feature_dim=self.backbone.feature_dim,
            modality=self.modality,
            dropout=self.dropout
        )

        # Set final feature dimension
        self.feature_dim = 512 if len(self.modality) > 1 else self.backbone.feature_dim
        self.fc = None  # Classifier will be created via update_fc()

        print("=" * 40)
        print("✅ Baseline Model Configuration")
        print("-" * 40)
        print(f"  Backbone:        {self.backbone_name}")
        print(f"  Fusion:          {self.midfusion}")
        print(f"  Modality:        {self.modality}")
        print(f"  Segments:        {self.num_segments}")
        print(f"  Dropout:         {self.dropout}")
        print(f"  Consensus:       {self.consensus_type}")
        print("=" * 40)

    @property
    def output_dim(self):
        """Return output feature dimension"""
        return self.feature_dim

    def extract_vector(self, x):
        """Extract fused features without classification"""
        features = self.backbone(x)  # Extract per-modality features
        fused = self.fusion(features)  # Fuse multi-modal features
        return fused["features"]

    def forward(self, x):
        """Forward pass: backbone -> fusion -> classifier"""
        features = self.backbone(x)  # Extract features from each modality
        fused = self.fusion(features)  # Combine features across modalities
        out = self.fc(fused["features"])  # Apply classifier
        out.update(fused)  # Include fusion output
        return out

    def update_fc(self, nb_classes):
        """Update classifier for new number of classes while preserving weights"""
        # Create new classifier with updated class count
        new_fc = ClassificationTBN(
            feature_dim=self.feature_dim,
            modality=self.modality,
            num_class=nb_classes,
            consensus_type=self.consensus_type,
            before_softmax=self.before_softmax,
            num_segments=self.num_segments
        )

        # Preserve existing classifier weights if available
        if self.fc is not None:
            nb_output = self.fc.num_class
            new_fc.fc_action.weight.data[:nb_output] = self.fc.fc_action.weight.data
            new_fc.fc_action.bias.data[:nb_output] = self.fc.fc_action.bias.data

        self.fc = new_fc

    def copy(self):
        """Create deep copy of the model"""
        return copy.deepcopy(self)

    def freeze(self):
        """Freeze all parameters for inference"""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self
