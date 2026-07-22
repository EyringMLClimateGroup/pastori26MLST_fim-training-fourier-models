"""
JAX/optax training loop and loss functions used to train the Fourier/tensor-network/QNN
regression models against a mean-squared-error (MSE) objective, and to track training and
validation loss (and, optionally, parameter trajectories) over epochs. These routines implement
the training experiments of the paper (Sec. on training and effective dimension), where models
are trained to fit a data-generating function y(x) and the minimum achieved MSE
(MSE_min, and its difference Delta_{f-c}MSE_min between a 'full' and a 'cutoff' model) is
compared against the models' effective dimension and bias.
"""

import jax
from jax import numpy as jnp
import optax

from sklearn.utils import shuffle





###
### MSE loss function for a model returning a single output.
###
### This function is to be wrapped anf JIT compiled in the main as follows:
###
### my_model = ...
###
### @jax.jit
### def loss(params, inputs, targets):
###     return training_functions_jax.mse_loss(params, inputs, targets, model=my_model)
###
def mse_loss(params, inputs, targets, model):
    """
    Mean-squared-error loss (1/no_samples) * sum_x (y(x) - f_theta(x))**2 for a model returning a
    single scalar output per input, used as the training objective whose minimum value MSE_min
    is compared across models of different effective dimension/bias in the paper's training
    experiments.

    Parameters
    ----------
    params : pytree / array
        Model parameters theta.
    inputs : ndarray, shape (batch_size, no_features)
        Batch of input points x.
    targets : ndarray, shape (batch_size,)
        Target/data-generating values y(x) for each input.
    model : callable
        Function model(params, inputs) -> predictions, shape (batch_size,).

    Returns
    -------
    loss : float
    """
    predictions = model(params, inputs)
    loss = jnp.sum((targets - predictions) ** 2.0 / len(targets))
    return loss








###
### MSE + alpha * VAR loss function for a model returning a single output.
### 'model' is supposed to return the model outputs together with their variance.
###
### Before being fed to the 'train_model' routine, this function is to be wrapped anf JIT compiled in the main as follows:
###
### alpha = ...
###
### @jax.jit
### def loss(params, inputs, targets, model_with_var):
###     return training_functions_jax.mse_plus_var_loss(params, inputs, targets, model_with_var, alpha=alpha)
###
def mse_plus_var_loss(params, inputs, targets, model, alpha):
    """
    Regularized loss MSE + alpha * mean(variance), for a model that returns both a prediction
    and an associated per-sample variance (e.g. an ensemble/stochastic model). The variance
    penalty term discourages high-variance predictions, with 'alpha' controlling its weight
    relative to the plain MSE loss (mse_loss).

    Parameters
    ----------
    params : pytree / array
        Model parameters theta.
    inputs : ndarray, shape (batch_size, no_features)
        Batch of input points x.
    targets : ndarray, shape (batch_size,)
        Target/data-generating values y(x) for each input.
    model : callable
        Function model(params, inputs) -> (predictions, variances), each shape (batch_size,).
    alpha : float
        Weight of the variance penalty term.

    Returns
    -------
    loss : float
    """
    no_data = len(targets)
    predictions, variances = model(params, inputs)
    mse_loss = jnp.sum((targets - predictions) ** 2.0 / no_data)
    var_loss = jnp.mean(variances)
    loss = mse_loss + alpha * var_loss
    return loss







###
### Define the function for training the model
###
### This function can be wrapped anf JIT compiled in the main as follows:
###
### opt = ...
### loss = ...
### no_epochs = ...
### batch_size = ...
###
### @jax.jit
### def train_model_jit(params):
###     args_opt = (opt, loss, no_epochs, batch_size)
###     opt_params, loss_history = training_functions_jax.train_model(args_opt, model_qnn, params, train_inputs, 
###                                                                   train_outputs, val_inputs, val_outputs)
###     return opt_params, loss_history
###
def train_model(args_opt, model, params, train_inputs, train_targets, val_inputs, val_targets):
    """
    Train 'model' by mini-batch gradient descent (using the optax optimizer 'opt') to minimize
    'loss' on (train_inputs, train_targets), for 'no_epochs' epochs with the given 'batch_size',
    printing the training and validation loss at the end of every epoch. The whole optimization
    loop is compiled with jax.lax.fori_loop for speed, so 'no_epochs', 'batch_size', and the
    dataset sizes must be static (known at compile time).

    Used to obtain the training-loss curves (and MSE_min at convergence) that are related, in
    the paper's experiments, to a model's effective dimension and bias towards the target
    function y(x).

    Parameters
    ----------
    args_opt : tuple (opt, loss, no_epochs, batch_size)
        opt : optax optimizer (e.g. optax.adam(...)).
        loss : callable loss(params, inputs, targets, model) -> scalar (e.g. mse_loss).
        no_epochs : int, number of training epochs.
        batch_size : int, mini-batch size.
    model : callable
        Function model(params, inputs) -> predictions.
    params : pytree / array
        Initial model parameters theta.
    train_inputs, train_targets : ndarray
        Training set inputs (batch, no_features) and targets (batch,).
    val_inputs, val_targets : ndarray
        Validation set inputs and targets, evaluated (not trained on) each epoch.

    Returns
    -------
    opt_params : pytree / array
        Trained parameters.
    loss_history : dict with keys 'train_loss', 'val_loss'
        Arrays of shape (no_epochs,) with the loss at the end of each epoch.
    """
    opt, loss, no_epochs, batch_size = args_opt
    
    trainset_size = train_inputs.shape[0]
    valset_size = val_inputs.shape[0]
    no_features = train_inputs.shape[1]
    no_batches_train = int(trainset_size / batch_size)
    no_batches_val = int(valset_size / batch_size)

    
    ### compile the loss function
    @jax.jit
    def loss_fn(params, inputs, targets):
        return loss(params, inputs, targets, model=model)

    
    ### compile the update step for one batch
    @jax.jit
    def update_step_batch_jit(i, args):
        params, opt_state, in_train, out_train = args
        ins = jax.lax.dynamic_slice(in_train, [i*batch_size, 0], [batch_size, no_features])
        outs = jax.lax.dynamic_slice(out_train, [i*batch_size], [batch_size])
        loss_val, grads = jax.value_and_grad(loss_fn)(params, ins, outs)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, in_train, out_train)


    ### compile function for calculating loss over one batch
    @jax.jit
    def calculate_dataset_loss_jit(i, args):
        loss_value, params, in_data, out_data = args
        ins = jax.lax.dynamic_slice(in_data, [i*batch_size, 0], [batch_size, no_features])
        outs = jax.lax.dynamic_slice(out_data, [i*batch_size], [batch_size])
        loss_batch = loss_fn(params, ins, outs)
        loss_value = loss_value + loss_batch
        return (loss_value, params, in_data, out_data)

    
    ### compile the update step for one epoch
    @jax.jit
    def update_step_epoch_jit(i, args):
        params, loss_hist, opt_state, in_train, out_train, in_val, out_val = args
        
        # shuffle training data and loop over batches
        inputs, outputs = shuffle(in_train, out_train)
        args_batch = (params, opt_state, inputs, outputs)
        (params, opt_state, _, _) = jax.lax.fori_loop(0, no_batches_train, update_step_batch_jit, args_batch)
        
        # calculate train and validation losses at the end of each epoch
        train_loss = 0.0; args_train_loss = (train_loss, params, in_train, out_train)
        (train_loss, _, _, _) = jax.lax.fori_loop(0, no_batches_train, calculate_dataset_loss_jit, args_train_loss)
        train_loss = train_loss / no_batches_train
        val_loss = 0.0; args_val_loss = (val_loss, params, in_val, out_val)
        (val_loss, _, _, _) = jax.lax.fori_loop(0, no_batches_val, calculate_dataset_loss_jit, args_val_loss)
        val_loss = val_loss / no_batches_val

        # update loss history container
        train_hist = loss_hist['train_loss']
        train_hist = train_hist.at[i].set(train_loss)
        loss_hist['train_loss'] = train_hist
        val_hist = loss_hist['val_loss']
        val_hist = val_hist.at[i].set(val_loss)
        loss_hist['val_loss'] = val_hist
        
        # Print the train and val loss at the end of each epoch
        jax.debug.print("Epoch {ep}  ----  Train loss: {train_loss}  ----  Val. loss: {val_loss}", ep=(i+1), train_loss=train_loss, val_loss=val_loss)
        
        return (params, loss_hist, opt_state, in_train, out_train, in_val, out_val)

    
    ### compile the optimization loop
    @jax.jit
    def optimization_jit(params, train_data, val_data):
        in_train, out_train = train_data
        in_val, out_val = val_data

        # initialize optimizer and loss history container
        opt_state = opt.init(params)
        loss_history = dict()
        loss_history['train_loss'] = jnp.zeros(no_epochs)
        loss_history['val_loss'] = jnp.zeros(no_epochs)

        # run optimization loop
        args = (params, loss_history, opt_state, in_train, out_train, in_val, out_val)
        (params, loss_history, opt_state, _, _, _, _) = jax.lax.fori_loop(0, no_epochs, update_step_epoch_jit, args)
    
        return params, loss_history

    
    ### run the optimization
    train_data = (train_inputs, train_targets)
    val_data = (val_inputs, val_targets)
    opt_params, loss_history = optimization_jit(params, train_data, val_data)
    
    return opt_params, loss_history








###
### Define the function for training the model
###
### This function can be wrapped anf JIT compiled in the main as follows:
###
### opt = ...
### loss = ...
### no_epochs = ...
### batch_size = ...
###
### @jax.jit
### def train_model_jit(params):
###     args_opt = (opt, loss, no_epochs, batch_size)
###     opt_params, loss_history = training_functions_jax.train_model(args_opt, model_qnn, params, train_inputs, 
###                                                                   train_outputs, val_inputs, val_outputs)
###     return opt_params, loss_history
###
def train_model_noprint(args_opt, model, params, train_inputs, train_targets, val_inputs, val_targets):
    """
    Identical to train_model, but without per-epoch print statements -- intended for use inside
    large parameter/data-scan loops (e.g. scanning bond dimension, sparsity, or bias) where
    epoch-by-epoch logging for every run would be excessive.

    Parameters
    ----------
    args_opt : tuple (opt, loss, no_epochs, batch_size)
        See train_model.
    model : callable
        Function model(params, inputs) -> predictions.
    params : pytree / array
        Initial model parameters theta.
    train_inputs, train_targets : ndarray
        Training set inputs and targets.
    val_inputs, val_targets : ndarray
        Validation set inputs and targets.

    Returns
    -------
    opt_params : pytree / array
        Trained parameters.
    loss_history : dict with keys 'train_loss', 'val_loss'
        Arrays of shape (no_epochs,) with the loss at the end of each epoch.
    """
    opt, loss, no_epochs, batch_size = args_opt
    
    trainset_size = train_inputs.shape[0]
    valset_size = val_inputs.shape[0]
    no_features = train_inputs.shape[1]
    no_batches_train = int(trainset_size / batch_size)
    no_batches_val = int(valset_size / batch_size)

    
    ### compile the loss function
    @jax.jit
    def loss_fn(params, inputs, targets):
        return loss(params, inputs, targets, model=model)

    
    ### compile the update step for one batch
    @jax.jit
    def update_step_batch_jit(i, args):
        params, opt_state, in_train, out_train = args
        ins = jax.lax.dynamic_slice(in_train, [i*batch_size, 0], [batch_size, no_features])
        outs = jax.lax.dynamic_slice(out_train, [i*batch_size], [batch_size])
        loss_val, grads = jax.value_and_grad(loss_fn)(params, ins, outs)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, in_train, out_train)


    ### compile function for calculating loss over one batch
    @jax.jit
    def calculate_dataset_loss_jit(i, args):
        loss_value, params, in_data, out_data = args
        ins = jax.lax.dynamic_slice(in_data, [i*batch_size, 0], [batch_size, no_features])
        outs = jax.lax.dynamic_slice(out_data, [i*batch_size], [batch_size])
        loss_batch = loss_fn(params, ins, outs)
        loss_value = loss_value + loss_batch
        return (loss_value, params, in_data, out_data)

    
    ### compile the update step for one epoch
    @jax.jit
    def update_step_epoch_jit(i, args):
        params, loss_hist, opt_state, in_train, out_train, in_val, out_val = args
        
        # shuffle training data and loop over batches
        inputs, outputs = shuffle(in_train, out_train)
        args_batch = (params, opt_state, inputs, outputs)
        (params, opt_state, _, _) = jax.lax.fori_loop(0, no_batches_train, update_step_batch_jit, args_batch)
        
        # calculate train and validation losses at the end of each epoch
        train_loss = 0.0; args_train_loss = (train_loss, params, in_train, out_train)
        (train_loss, _, _, _) = jax.lax.fori_loop(0, no_batches_train, calculate_dataset_loss_jit, args_train_loss)
        train_loss = train_loss / no_batches_train
        val_loss = 0.0; args_val_loss = (val_loss, params, in_val, out_val)
        (val_loss, _, _, _) = jax.lax.fori_loop(0, no_batches_val, calculate_dataset_loss_jit, args_val_loss)
        val_loss = val_loss / no_batches_val

        # update loss history container
        train_hist = loss_hist['train_loss']
        train_hist = train_hist.at[i].set(train_loss)
        loss_hist['train_loss'] = train_hist
        val_hist = loss_hist['val_loss']
        val_hist = val_hist.at[i].set(val_loss)
        loss_hist['val_loss'] = val_hist
        
        # Print the train and val loss at the end of each epoch
        #jax.debug.print("Epoch {ep}  ----  Train loss: {train_loss}  ----  Val. loss: {val_loss}", ep=(i+1), train_loss=train_loss, val_loss=val_loss)
        
        return (params, loss_hist, opt_state, in_train, out_train, in_val, out_val)

    
    ### compile the optimization loop
    @jax.jit
    def optimization_jit(params, train_data, val_data):
        in_train, out_train = train_data
        in_val, out_val = val_data

        # initialize optimizer and loss history container
        opt_state = opt.init(params)
        loss_history = dict()
        loss_history['train_loss'] = jnp.zeros(no_epochs)
        loss_history['val_loss'] = jnp.zeros(no_epochs)

        # run optimization loop
        args = (params, loss_history, opt_state, in_train, out_train, in_val, out_val)
        (params, loss_history, opt_state, _, _, _, _) = jax.lax.fori_loop(0, no_epochs, update_step_epoch_jit, args)
    
        return params, loss_history

    
    ### run the optimization
    train_data = (train_inputs, train_targets)
    val_data = (val_inputs, val_targets)
    opt_params, loss_history = optimization_jit(params, train_data, val_data)
    
    return opt_params, loss_history








###
### Define the function for training the model and saving the weights training history.
###
### This function can be wrapped anf JIT compiled in the main as follows:
###
### opt = ...
### loss = ...
### no_epochs = ...
### batch_size = ...
### no_params = ...
###
### @jax.jit
### def train_model_weightsevo_jit(params):
###     args_opt = (opt, loss, no_epochs, batch_size, no_params, every_n_batches)
###     opt_params, loss_history, weights_history = training_functions_jax.train_model_weightsevo(args_opt, model_qnn, params, train_inputs, 
###                                                                                               train_outputs, val_inputs, val_outputs)
###     return opt_params, loss_history, weights_history
###
def train_model_weightsevo(args_opt, model, params, train_inputs, train_targets, val_inputs, val_targets):
    """
    Same training loop as train_model, but additionally snapshots the full parameter vector
    theta every 'every_n_batches' mini-batches (plus the initial parameters), returning the full
    weight trajectory. Used when the evolution of theta itself during training needs to be
    inspected or post-processed (e.g. to evaluate the FIM/effective dimension along the
    optimization trajectory), rather than only the final trained parameters and loss curve.

    Parameters
    ----------
    args_opt : tuple (opt, loss, no_epochs, batch_size, no_params, every_n_batches)
        opt, loss, no_epochs, batch_size : see train_model.
        no_params : int, number of trainable parameters M (length of theta).
        every_n_batches : int, save the parameter vector every this many mini-batches.
    model : callable
        Function model(params, inputs) -> predictions.
    params : pytree / array
        Initial model parameters theta.
    train_inputs, train_targets : ndarray
        Training set inputs and targets.
    val_inputs, val_targets : ndarray
        Validation set inputs and targets.

    Returns
    -------
    opt_params : pytree / array
        Trained parameters.
    loss_history : dict with keys 'train_loss', 'val_loss'
        Arrays of shape (no_epochs,) with the loss at the end of each epoch.
    weights_history : ndarray, shape (no_wghs_saves, no_params)
        Snapshots of theta taken every 'every_n_batches' batches throughout training (including
        the initial parameters as the first row).
    """
    opt, loss, no_epochs, batch_size, no_params, every_n_batches = args_opt
    
    trainset_size = train_inputs.shape[0]
    valset_size = val_inputs.shape[0]
    no_features = train_inputs.shape[1]
    no_batches_train = int(trainset_size / batch_size)
    no_batches_val = int(valset_size / batch_size)

    # No. of times per epoch when weights are saved
    no_wghs_saves_per_epoch = int(no_batches_train / every_n_batches)
    # Total no. of times when weights are saved (the '+1' is for the initial parameter values)
    no_wghs_saves = no_wghs_saves_per_epoch * no_epochs + 1

    
    ### compile the loss function
    @jax.jit
    def loss_fn(params, inputs, targets):
        return loss(params, inputs, targets, model=model)

    
    ### compile the update step for one batch
    @jax.jit
    def update_step_batch_jit(i, args):
        params, opt_state, weights_history, save_counter, in_train, out_train = args
        ins = jax.lax.dynamic_slice(in_train, [i*batch_size, 0], [batch_size, no_features])
        outs = jax.lax.dynamic_slice(out_train, [i*batch_size], [batch_size])
        loss_val, grads = jax.value_and_grad(loss_fn)(params, ins, outs)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        def update_weights_hist(args_wghs):
            weights_history, save_counter = args_wghs
            save_counter = save_counter + 1
            weights_history = weights_history.at[save_counter, :].set(params)
            return weights_history, save_counter
        def not_update_weights_hist(args_wghs):
            weights_history, save_counter = args_wghs
            return weights_history, save_counter
        args_wghs_fn = (weights_history, save_counter)
        weights_history, save_counter = jax.lax.cond(((i + 1) % every_n_batches)==0, update_weights_hist, not_update_weights_hist, args_wghs_fn)
        return (params, opt_state, weights_history, save_counter, in_train, out_train)


    ### compile function for calculating loss over one batch
    @jax.jit
    def calculate_dataset_loss_jit(i, args):
        loss_value, params, in_data, out_data = args
        ins = jax.lax.dynamic_slice(in_data, [i*batch_size, 0], [batch_size, no_features])
        outs = jax.lax.dynamic_slice(out_data, [i*batch_size], [batch_size])
        loss_batch = loss_fn(params, ins, outs)
        loss_value = loss_value + loss_batch
        return (loss_value, params, in_data, out_data)

    
    ### compile the update step for one epoch
    @jax.jit
    def update_step_epoch_jit(i, args):
        params, loss_hist, opt_state, weights_history, save_counter, in_train, out_train, in_val, out_val = args
        
        # shuffle training data and loop over batches
        inputs, outputs = shuffle(in_train, out_train)
        args_batch = (params, opt_state, weights_history, save_counter, inputs, outputs)
        (params, opt_state, weights_history, save_counter, _, _) = jax.lax.fori_loop(0, no_batches_train, update_step_batch_jit, args_batch)
        
        # calculate train and validation losses at the end of each epoch
        train_loss = 0.0; args_train_loss = (train_loss, params, in_train, out_train)
        (train_loss, _, _, _) = jax.lax.fori_loop(0, no_batches_train, calculate_dataset_loss_jit, args_train_loss)
        train_loss = train_loss / no_batches_train
        val_loss = 0.0; args_val_loss = (val_loss, params, in_val, out_val)
        (val_loss, _, _, _) = jax.lax.fori_loop(0, no_batches_val, calculate_dataset_loss_jit, args_val_loss)
        val_loss = val_loss / no_batches_val

        # update loss history container
        train_hist = loss_hist['train_loss']
        train_hist = train_hist.at[i].set(train_loss)
        loss_hist['train_loss'] = train_hist
        val_hist = loss_hist['val_loss']
        val_hist = val_hist.at[i].set(val_loss)
        loss_hist['val_loss'] = val_hist
        
        # Print the train and val loss at the end of each epoch
        jax.debug.print("Epoch {ep}  ----  Train loss: {train_loss}  ----  Val. loss: {val_loss}", ep=(i+1), train_loss=train_loss, val_loss=val_loss)
        
        return (params, loss_hist, opt_state, weights_history, save_counter, in_train, out_train, in_val, out_val)

    
    ### compile the optimization loop
    @jax.jit
    def optimization_jit(params, train_data, val_data):
        in_train, out_train = train_data
        in_val, out_val = val_data

        # initialize optimizer and loss history container
        opt_state = opt.init(params)
        loss_history = dict()
        loss_history['train_loss'] = jnp.zeros(no_epochs)
        loss_history['val_loss'] = jnp.zeros(no_epochs)

        # initialize weights history container
        weights_history = jnp.zeros((no_wghs_saves, no_params))
        save_counter = 0
        weights_history = weights_history.at[save_counter, :].set(params)

        # run optimization loop
        args = (params, loss_history, opt_state, weights_history, save_counter, in_train, out_train, in_val, out_val)
        (params, loss_history, opt_state, weights_history, save_counter, _, _, _, _) = jax.lax.fori_loop(0, no_epochs, update_step_epoch_jit, args)
    
        return params, loss_history, weights_history

    
    ### run the optimization
    train_data = (train_inputs, train_targets)
    val_data = (val_inputs, val_targets)
    opt_params, loss_history, weights_history = optimization_jit(params, train_data, val_data)
    
    return opt_params, loss_history, weights_history
