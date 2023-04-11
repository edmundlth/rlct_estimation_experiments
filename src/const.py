import jax

ACTIVATION_FUNC_SWITCH = {
    "tanh": jax.nn.tanh, 
    "id": lambda x: x, 
    "relu": jax.nn.relu, 
    "gelu": jax.nn.gelu, 
    "swish": jax.nn.swish, 
}
