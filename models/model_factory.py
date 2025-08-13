# ====================
# Model Factory Method
# ====================

# Import available models
from models.myewc import TBN_EWC, TSN_EWC
from models.mylwf import TBN_LwF, TSN_LwF
from models.myicarl import TBN_iCaRL, TSN_iCaRL
from models.cmr_mfn import CMR_MFN


def get_model(model_name, args):
    """Return model instance by name."""

    name = model_name.lower()

    # Dictionary mapping model names to their classes
    model_dict = {
        "tbn_ewc": TBN_EWC,
        "tsn_ewc": TSN_EWC,
        "tbn_lwf": TBN_LwF,
        "tsn_lwf": TSN_LwF,
        "tbn_icarl": TBN_iCaRL,
        "tsn_icarl": TSN_iCaRL,
        'cmr_mfn': CMR_MFN,
    }

    # Instantiate and return the model if name is valid
    if name in model_dict:
        return model_dict[name](args)  # Create and return model instance
    else:
        raise ValueError(f"Unknown model name: {model_name}")
