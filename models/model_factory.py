# ====================
# Model Factory Method
# ====================

# Import available models
from models.myewc import MyEWC
from models.cmr_mfn import CMR_MFN
from models.myicarl import MyiCaRL
from models.mylwf import MyLwF

def get_model(model_name, args):
    """Return model instance by name."""

    name = model_name.lower()

    # Dictionary mapping model names to their classes
    model_dict = {
        "myewc": MyEWC,
        'cmr_mfn': CMR_MFN,
        'myicarl': MyiCaRL,
        'mylwf': MyLwF
    }

    # Instantiate and return the model if name is valid
    if name in model_dict:
        return model_dict[name](args)  # Create and return model instance
    else:
        raise ValueError(f"Unknown model name: {model_name}")
