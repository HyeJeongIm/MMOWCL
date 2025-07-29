# ====================
# Model Factory Method
# ====================

# Import available models
from models.myewc import MyEWC

def get_model(model_name, args):
    """Return model instance by name."""

    name = model_name.lower()

    # Dictionary mapping model names to their classes
    model_dict = {
        "myewc": MyEWC,
    }

    # Instantiate and return the model if name is valid
    if name in model_dict:
        return model_dict[name](args)  # Create and return model instance
    else:
        raise ValueError(f"Unknown model name: {model_name}")
