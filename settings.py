import os
import numpy
import pathlib


def get_boolean_env(env_var: str, default: bool = False) -> bool:
    """
    Env vars are always strings
    Convert to boolean
    """
    val = os.environ.get(env_var, "").lower()
    if val in ["true", "1", "yes"]:
        return True
    elif val in ["false", "0", "no", "none"]:
        return False
    else:
        return default

# Pull some environment vars
STORE_PATH = os.environ.get("STORE_PATH", f"{pathlib.Path(__file__).parent.resolve()}/store")

# Toggle our Keras NN model to be Sequential or Mixed Data
KERAS_SEQUENTIAL = get_boolean_env("KERAS_SEQUENTIAL", False)
USE_GPU = get_boolean_env("USE_GPU", False)
USE_GENERATORS = get_boolean_env("USE_GENERATORS", True)

# Random seed
SEED = int(os.environ.get("SEED", 8675309))
numpy.random.seed(SEED)

# If running as a container, we don't want to wait for inputs
AWS_KEY_ID = os.environ.get("RR_AWS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("RR_AWS_ACCESS_KEY")

