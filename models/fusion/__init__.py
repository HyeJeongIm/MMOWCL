from .fusion_concat import FusionConcat
from .fusion_concat import FusionConcat
from .fusion_cmr import FusionCMR
# from .fusion_context_gating import FusionContextGating
# from .fusion_multimodal_gating import FusionMultimodalGating

def get_fusion(midfusion, feature_dim, modality, dropout, num_segments=None):
    if midfusion == "concat":
        return FusionConcat(feature_dim, modality, dropout)
    elif midfusion == "attention":
        # FusionCMR은 attention과 concat 모두 지원
        if num_segments is None:
            raise ValueError("num_segments is required for attention fusion")
        return FusionCMR(
            input_dim=feature_dim,
            modality=modality,
            fusion_type="attention",
            dropout=dropout,
            num_segments=num_segments
        )
    # elif midfusion == "context_gating":
    #     return FusionContextGating(feature_dim, modality, dropout)
    # elif midfusion == "multimodal_gating":
    #     return FusionMultimodalGating(feature_dim, modality, dropout)
    else:
        raise ValueError(f"Unknown midfusion type: {midfusion}")
