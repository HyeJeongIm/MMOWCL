# models/backbones/__init__.py
from models.backbones.tbn import TBN
# from models.backbones.timesformer_backbone import TimeSformerBackbone

def get_backbone(args):
    name = args["backbone"].lower()
    if name == "tbn":
        return TBN(
            num_segments=args["num_segments"],
            modality=args["modality"],
            base_model=args["arch"],
            new_length=args.get("new_length", None)
            )
    # elif name == "timesformer":
    #     return TimeSformerBackbone(
    #         ...  # 구현 예정 => Multi-Modal 방법론 B가 되는 부분 
    #     )
    else:
        raise ValueError(f"Unknown backbone: {name}")
