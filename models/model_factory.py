# ====================
# Model Factory Method
# ====================

# Import available models
from models.myewc import TBNEWC, TSNEWC
from models.mylwf import TBNLwF, TSNLwF
from models.myicarl import TBNiCaRL, TSNiCaRL
from models.cmr_mfn import CMR_MFN


def get_model(model_name, args):
    """Return model instance by name."""

    name = model_name.lower()

    # Dictionary mapping model names to their classes
    model_dict = {
        "tbnewc": TBNEWC,
        "tsnewc": TSNEWC,
        "tbnlwf": TBNLwF,
        "tsnlwf": TSNLwF,
        "tbnicarl": TBNiCaRL,
        "tsnicarl": TSNiCaRL,
        'cmr_mfn': CMR_MFN,
    }

    # Instantiate and return the model if name is valid
    if name in model_dict:
        return model_dict[name](args)  # Create and return model instance
    else:
        raise ValueError(f"Unknown model name: {model_name}")
