from .fusion_concat import FusionConcat
# from .fusion_context_gating import FusionContextGating
# from .fusion_multimodal_gating import FusionMultimodalGating

def get_fusion(midfusion, feature_dim, modality, dropout):
    if midfusion == "concat":
        return FusionConcat(feature_dim, modality, dropout)
    # elif midfusion == "context_gating":
    #     return FusionContextGating(feature_dim, modality, dropout)
    # elif midfusion == "multimodal_gating":
    #     return FusionMultimodalGating(feature_dim, modality, dropout)
    else:
        raise ValueError(f"Unknown midfusion type: {midfusion}")
