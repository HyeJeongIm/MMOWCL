# models/baseline_cmr.py
# B버전 (CMR_MFN, TSN) 전용 Baseline Network

import torch
import torch.nn as nn
import copy
import logging

from models.backbones import get_backbone
from models.fusion import get_fusion

from models.fusion.fusion_cmr import FusionCMR
from models.classifier.classification_cmr import ClassificationCMR


class BaselineCMR(nn.Module):
    """
    B버전 (CMR_MFN, TSN) 전용 Multi-modal baseline network
    - gen_train_fc 방식 사용
    - fusion_networks, fc_list로 태스크별 파라미터 저장
    - CMR_MFNNet과 동일한 구조
    """
    
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_segments = args["num_segments"]
        self.modality = args["modality"]
        self.backbone_name = args["backbone"]  # e.g., 'tsn'
        self.fusion_type = args["fusion_type"]  # e.g., 'attention', 'concat'
        self.dropout = args["dropout"]

        # B버전: 태스크별 파라미터 저장을 위한 리스트
        self.fusion_networks = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        self.fc = None  # Current classifier

        # Initialize backbone network (TSN)
        self.backbone = get_backbone(args)  # TSN 백본
        
        # Initialize fusion network (CMR 방식)
        self.fusion_network = get_fusion(
            midfusion=self.fusion_type,  # 'attention' or 'concat'
            feature_dim=768,  # ViT feature dimension
            modality=self.modality,
            dropout=self.dropout,
            num_segments=self.num_segments
        )

        # Set feature dimension for classifier
        if len(self.modality) > 1:
            if self.fusion_type == 'attention':
                self.feature_dim = 128
            elif self.fusion_type == 'concat':
                self.feature_dim = 768 * len(self.modality)
        else:
            self.feature_dim = 768

        logging.info(f"""
Initializing BaselineCMR with backbone: {self.backbone_name}.
CMR Configurations:
    input_modality:     {self.modality}
    num_segments:       {self.num_segments}
    fusion_type:        {self.fusion_type}
    dropout_ratio:      {self.dropout}
    feature_dim:        {self.feature_dim}
""")

    @property
    def output_dim(self):
        """Return output feature dimension"""
        return self.feature_dim

    def extract_vector(self, x):
        """Extract fused features without classification"""
        # Extract features from backbone (TSN)
        features = self.backbone(x)
        # Fuse features using CMR fusion
        fused = self.fusion_network(features)
        return fused["features"]

    def forward(self, x, cur_task_size=None, mode='train'):
        # Extract features from backbone
        features = self.backbone(x)

        if mode == 'train':
            fusion_output = self.fusion_network(features)
            fusion_features = fusion_output["features"]
            out = self.fc(fusion_features)
            out.update({"features": features, "fusion_features": fusion_features})
            
        elif mode == 'test':
            fusion_features_list, logits_list = [], []
            
            for idx, fc in enumerate(self.fc_list):
                fusion_output = self.fusion_networks[idx](features)
                fusion_feat = fusion_output["features"]
                logits = fc(fusion_feat)["logits"]
                
                if cur_task_size is not None:
                    logits = logits[:, :cur_task_size]
                    
                fusion_features_list.append(fusion_feat)
                logits_list.append(logits)

            # Concatenate results from all tasks
            fusion_features = torch.cat(fusion_features_list, 1)
            logits = torch.cat(logits_list, 1)
            out = {"logits": logits}
            out.update({"features": features, "fusion_features": fusion_features})

        return out

    def gen_train_fc(self, incre_classes):
        """
        B버전: gen_train_fc 방식으로 classifier 생성/확장
        """
        # Create new classifier
        new_fc = ClassificationCMR(
            feature_dim=self.feature_dim,
            modality=self.modality,
            num_class=incre_classes,
            dropout=self.dropout,
            num_segments=self.num_segments
        )

        # Copy weights from existing classifier if available
        if self.fc is not None:
            num_old_classes = self.fc.num_class
            new_fc.copy_weights_from(self.fc, num_old_classes)

        # Replace current classifier
        del self.fc
        self.fc = new_fc

    def save_parameter(self):
        """
        현재 태스크의 파라미터를 리스트에 저장
        """
        # Save current fusion network
        new_fusion = get_fusion(
            midfusion=self.fusion_type,
            feature_dim=768,
            modality=self.modality,
            dropout=self.dropout,
            num_segments=self.num_segments
        )
        new_fusion.load_state_dict(self.fusion_network.state_dict())
        self.fusion_networks.append(new_fusion)

        # Save current classifier
        if self.fc is not None:
            new_fc = ClassificationCMR(
                feature_dim=self.feature_dim,
                modality=self.modality,
                num_class=self.fc.num_class,
                dropout=self.dropout,
                num_segments=self.num_segments
            )
            new_fc.load_state_dict(self.fc.state_dict())
            self.fc_list.append(new_fc)

    def copy(self):
        """Create deep copy of the model"""
        return copy.deepcopy(self)

    def freeze(self):
        """Freeze all parameters for inference"""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self