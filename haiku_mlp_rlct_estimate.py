import argparse
import os

import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import scipy
import functools

import haiku as hk
import numpyro
import matplotlib.pyplot as plt

from src.haiku_numpyro_mlp import (
    build_forward_fn, 
    build_log_likelihood_fn, 
    generate_input_data, 
    generate_output_data, 
    build_model, 
    run_mcmc, 
    expected_nll
)
from src.const import ACTIVATION_FUNC_SWITCH
from src.utils import start_log

def main(args):
    logger = start_log()

    rngseed = args.rng_seed
    rngkeyseq = hk.PRNGSequence(jax.random.PRNGKey(rngseed))
    if args.layer_sizes is None:
        true_param_array = [jnp.array([list(args.b0)]), jnp.array([list(args.a0)])]
        layer_sizes = [len(args.b0), len(args.a0)]
    else:
        true_param_array = None
        layer_sizes = args.layer_sizes

    input_dim = args.input_dim

    activation_fn = ACTIVATION_FUNC_SWITCH[args.activation_fn_name.lower()]
    forward = build_forward_fn(
        layer_sizes=layer_sizes,
        activation_fn=activation_fn,
        initialisation_mean=args.prior_mean,
        initialisation_std=args.prior_std,
    )
    forward = hk.transform(forward)
    log_likelihood_fn = functools.partial(
        build_log_likelihood_fn, forward.apply, sigma=args.sigma_obs
    )

    X = generate_input_data(
        args.num_training_data, input_dim=input_dim, rng_key=next(rngkeyseq)
    )
    init_param = forward.init(next(rngkeyseq), X)
    init_param_flat, treedef = jtree.tree_flatten(init_param)
    # param_shapes = [p.shape for p in init_param_flat]
    if true_param_array is not None:
        true_param = treedef.unflatten(true_param_array)
    else:
        true_param = init_param
    Y = generate_output_data(
        forward, true_param, X, next(rngkeyseq), sigma=args.sigma_obs
    )
    param_center = true_param

    # param_prior_sampler = functools.partial(
    #     const_factorised_normal_prior, param_shapes, treedef, args.prior_mean, args.prior_std
    # )
    model = functools.partial(build_model, forward.apply)

    n = X.shape[0]
    itemps = jnp.linspace(
        1 / jnp.log(n) * (1 - 1 / jnp.sqrt(2 * jnp.log(n))),
        1 / jnp.log(n) * (1 + 1 / jnp.sqrt(2 * jnp.log(n))),
        args.num_itemps,
    )
    logger.info(f"itemps={itemps}")

    enlls = []
    for i_itemp, itemp in enumerate(itemps):
        mcmc = run_mcmc(
            model,
            X,
            Y,
            next(rngkeyseq),
            param_center,
            args.prior_mean,
            args.prior_std,
            args.sigma_obs,
            num_posterior_samples=args.num_posterior_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            thinning=args.thinning,
            itemp=itemp,
            progress_bar=(not args.quiet),
        )
        posterior_samples = mcmc.get_samples()
        num_mcmc_samples = num_mcmc_samples = len(
            posterior_samples[list(posterior_samples.keys())[0]]
        )
        param_list = [
            [posterior_samples[name][i] for name in sorted(posterior_samples.keys())]
            for i in range(num_mcmc_samples)
        ]
        enll = expected_nll(log_likelihood_fn, map(treedef.unflatten, param_list), X, Y)
        enlls.append(enll)
        logger.info(f"Finished {i_itemp} temp={1/itemp:.3f}. Expected NLL={enll:.3f}")
        if len(enlls) > 1:
            slope, intercept, r_val, _, _ = scipy.stats.linregress(
                1 / itemps[: len(enlls)], enlls
            )
            logger.info(
                f"est. RLCT={slope:.3f}, energy={intercept / n:.3f}, r2={r_val**2:.3f}"
            )

    fig, ax = plt.subplots(figsize=(5, 5))
    slope, intercept, r_val, _, _ = scipy.stats.linregress(1 / itemps, enlls)
    ax.plot(1 / itemps, enlls, "kx")
    ax.plot(
        1 / itemps,
        1 / itemps * slope + intercept,
        label=f"$\lambda$={slope:.3f}, $nL_n(w_0)$={intercept:.3f}, $R^2$={r_val**2:.2f}",
    )
    ax.legend()
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Expected NLL")
    n = args.num_training_data
    ax.set_title(f"n={n}, L_n={intercept / n:.3f}")
    if args.show_plot:
        plt.show()
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        fig.savefig(os.path.join(args.output_dir, "plot.png"))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLCT estimation of MLP models.")
    parser.add_argument("--num-itemps", nargs="?", default=6, type=int, help="Number of inverse temperature values (centered aroudn 1/log(n)) to be used in the regression for RLCT estimation.")
    parser.add_argument("--num-posterior-samples", nargs="?", default=2000, type=int, help="Number of posterior samples drawn in each MCMC chain.")
    parser.add_argument("--thinning", nargs="?", default=2, type=int, help="Thinning factor for MCMC sampling.")
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int, help="Number of samples discarded at the begining of each MCMC chain, a.k.a. burn in. ")
    parser.add_argument("--num-chains", nargs="?", default=4, type=int, help="Number of independent MCMC chains to run. This will set the number of compute devices used as well.")
    parser.add_argument("--num-training-data", nargs="?", default=1032, type=int, help="Size of the randomly generated training data set.")
    parser.add_argument("--a0", nargs="+", default=[0.5], type=float)
    parser.add_argument("--b0", nargs="+", default=[0.9], type=float)
    parser.add_argument("--input_dim", nargs="?", default=1, type=int, help="Dimension of the input data X.")
    parser.add_argument(
        "--layer_sizes",
        nargs="+",
        default=None,
        type=int,
        help="A optional list of positive integers specifying MLP layers sizes from the first non-input layer up to and including the output layer. If specified, --a0 and --b0 are ignored. ",
    )
    parser.add_argument("--sigma-obs", nargs="?", default=0.1, type=float)
    parser.add_argument("--prior-std", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior-mean", nargs="?", default=0.0, type=float)
    parser.add_argument("--activation-fn-name", nargs="?", default="tanh", type=str)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="a directory for storing output files.",
    )
    parser.add_argument(
        "--show_plot", action="store_true", default=True, help="Do plt.show()"
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False, help="Lower verbosity level."
    )
    parser.add_argument("--rng-seed", nargs="?", default=42, type=int)

    args = parser.parse_args()
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
