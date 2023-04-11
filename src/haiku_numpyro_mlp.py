import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import numpy as np
import optax
import haiku as hk
import numpyro
import numpyro.distributions as dist
import os
import time

import logging
logger = logging.getLogger(__name__)

def const_factorised_normal_prior(param_example, prior_mean=0.0, prior_std=1.0):
    """
    Return a param PyTree with the same structure as `param_example` but with every
    element replaced with a random sample from normal distribution with `prior_mean` and `prior_std`.
    """
    param_flat, treedef = jtree.tree_flatten(param_example)
    result = []
    for i, param in enumerate(param_flat):
        result.append(
            numpyro.sample(
                str(i),
                dist.Normal(loc=prior_mean, scale=prior_std),
                sample_shape=param.shape,
            )
        )
    return treedef.unflatten(result)


def localised_normal_prior(param_center, std=1.0):
    """
    Return a param PyTree with the same structure as `param_center` but with every
    element replaced with a random sample from normal distribution centered around values of `param_center` with standard deviation `std`.
    """
    result = []
    param_flat, treedef = jtree.tree_flatten(param_center)
    for i, p in enumerate(param_flat):
        result.append(numpyro.sample(str(i), dist.Normal(loc=p, scale=std)))
    return treedef.unflatten(result)


def build_forward_fn(
    layer_sizes, activation_fn, initialisation_mean=0.0, initialisation_std=1.0, with_bias=False
):
    """
    Construct a Haiku transformed forward function for an MLP network
    based on specified architectural parameters.
    """
    w_initialiser = hk.initializers.RandomNormal(
        stddev=initialisation_std, mean=initialisation_mean
    )

    def forward(x):
        mlp = hk.nets.MLP(
            layer_sizes, activation=activation_fn, w_init=w_initialiser, with_bias=with_bias
        )
        return mlp(x)

    return forward


def build_loss_fn(forward_fn, param, x, y):
    y_pred = forward_fn(param, None, x)
    return jnp.mean(optax.l2_loss(y_pred, y))


def build_model(
    forward_fn, X, Y, param_center, prior_mean, prior_std, itemp=1.0, sigma=1.0
):
    param_dict = const_factorised_normal_prior(param_center, prior_mean, prior_std)
    # param_dict = localised_normal_prior(param_center, prior_std)
    y_hat = forward_fn(param_dict, None, X)
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample(
            "Y", dist.Normal(y_hat, sigma / jnp.sqrt(itemp)).to_event(1), obs=Y
        )
    return


def build_log_likelihood_fn(forward_fn, param, x, y, sigma=1.0):
    y_hat = forward_fn(param, None, x)
    ydist = dist.Normal(y_hat, sigma)
    return ydist.log_prob(y).sum()


def expected_nll(log_likelihood_fn, param_list, X, Y):
    nlls = []
    for param in param_list:
        nlls.append(-log_likelihood_fn(param, X, Y))
    return np.mean(nlls)


def generate_input_data(num_training_data, input_dim, rng_key, xmin=-2, xmax=2):
    X = jax.random.uniform(
        key=rng_key,
        shape=(num_training_data, input_dim),
        minval=xmin,
        maxval=xmax,
    )
    return X


def generate_output_data(foward_fn, param, X, rng_key, sigma=0.1):
    y_true = foward_fn.apply(param, None, X)
    Y = y_true + jax.random.normal(rng_key, y_true.shape) * sigma
    return Y


def run_mcmc(
    model,
    X,
    Y,
    rng_key,
    param_center,
    prior_mean,
    prior_std,
    sigma,
    num_posterior_samples=2000,
    num_warmup=1000,
    num_chains=1,
    thinning=1,
    itemp=1.0,
    progress_bar=True,
):
    kernel = numpyro.infer.NUTS(model)
    logger.info("Running MCMC...")
    start_time = time.time()
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_samples=num_posterior_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        thinning=thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else progress_bar,
    )
    mcmc.run(
        rng_key, X, Y, param_center, prior_mean, prior_std, itemp=itemp, sigma=sigma
    )
    logger.info(f"Finished running MCMC. Time taken: {time.time() - start_time:.3f} seconds")
    return mcmc

