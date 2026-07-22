"""
JAX auto-differentiation route to the empirical Fisher Information Matrix (FIM) of a
regression model f_theta(x), F_{j,k}(theta) = E_x[df_theta/dtheta_j * df_theta/dtheta_k]
(Eq. (11) of the paper). Unlike the analytical/tensor-network FIM routines elsewhere in this
package (which use the known Fourier structure Gamma = U @ diag(S) @ Vh of the model), this
module treats the model as a black box and obtains the FIM by automatic differentiation and
Monte-Carlo averaging over a batch of input samples. It is the general-purpose FIM estimator
used for models (e.g. quantum neural networks or deep tensorized models) for which the
structure-constants decomposition is not directly available or convenient to use.
"""

import jax
from jax import numpy as jnp





###
### Routine for computing FIM for single-output regression
###
### This function is to be wrapped anf JIT compiled in the main as follows:
###
### my_model = ...
###
### @jax.jit
### def FIM_regression_jit(params, inputs):
###     return FIM_functions_jax.FIM_regression(params, inputs, model=my_model)
###
def FIM_regression(params, inputs, model):
    """
    Empirical Fisher Information Matrix for a single-output regression model, estimated by
    Monte-Carlo averaging over the given batch of inputs:
    F(theta) ~= (1/no_samples) * sum_x grad_theta(f_theta(x)) outer grad_theta(f_theta(x)).

    'model' is any JAX-differentiable callable model(params, x) -> scalar output (e.g. a
    Fourier/tensor-network model or a QNN expectation value); gradients are obtained via
    jax.grad and vectorized over the batch with jax.vmap, so no analytical knowledge of the
    model's structure constants Gamma is required.

    Meant to be wrapped and JIT-compiled as:

        my_model = ...

        @jax.jit
        def FIM_regression_jit(params, inputs):
            return FIM_functions_jax.FIM_regression(params, inputs, model=my_model)

    Parameters
    ----------
    params : pytree / array, shape (no_weights,)
        Model parameters theta at which the FIM is evaluated.
    inputs : ndarray, shape (no_samples, no_features)
        Batch of input points x used to estimate the expectation over the input distribution.
    model : callable
        Function model(params, x) -> scalar prediction for a single input x.

    Returns
    -------
    FIM : ndarray, shape (no_weights, no_weights)
        Empirical Fisher Information Matrix F(theta).
    """
    # compute gradients of the model output w.r.t. the weights
    grads = jax.vmap(jax.grad(model, argnums=0), in_axes=(None, 0))(params, inputs)  # dimension (no_samples, no_weights)

    # sample-wise outer product to compute the FIM for each sample
    gv = jnp.expand_dims(grads, axis=2)  # dimension (no_samples, no_weights, 1)
    gv_transpose = jnp.transpose(gv, axes=[0, 2, 1])  # dimension (no_samples, 1, no_weights)
    FIM_elems = jnp.multiply(gv, gv_transpose)  # dimension (no_samples, no_weights, no_weights)

    # empirical FIM is the mean over the input samples
    FIM = jnp.mean(FIM_elems, axis=0)  # dimension (no_weights, no_weights)
    
    return FIM
