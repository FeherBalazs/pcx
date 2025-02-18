# MNIST Classification with Predictive Coding Networks

import jax
import jax.numpy as jnp
import equinox as eqx
import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.functional as pxf
import pcx.utils as pxu
import jax.random

import optax
from typing import Callable
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

# Load MNIST using scikit-learn
def get_datasets(batch_size: int):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype('float32') / 255.0
    y = y.astype('int32')
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create batch indices
    def batch_indices(length, batch_size):
        indices = np.arange(length)
        np.random.shuffle(indices)
        n_batches = length // batch_size
        # Only keep indices that form full batches.
        return np.array_split(indices[: n_batches * batch_size], n_batches)
    
    train_batches = [(X_train[idx], y_train[idx]) for idx in batch_indices(len(X_train), batch_size)]
    test_batches = [(X_test[idx], y_test[idx]) for idx in batch_indices(len(X_test), batch_size)]
    
    return train_batches, test_batches

# Custom linear layer with explicit weight transposition (like PyTorch)
class MyLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_features: int, out_features: int, *, key):
        w_key, b_key = jax.random.split(key)
        # Use variance scaling (Xavier-like scaling) for stable activation magnitudes
        self.weight = jax.random.normal(w_key, (out_features, in_features)) / jnp.sqrt(in_features)
        self.bias = 0.01 * jax.random.normal(b_key, (out_features,))

    def __call__(self, x):
        # x dot weight.T: shape (..., in_features) dot (in_features, out_features) = (..., out_features)
        return jnp.dot(x, self.weight.T) + self.bias

# Define the model
class MNISTModel(pxc.EnergyModule):
    def __init__(
        self,
        hidden_dims: list[int],
        act_fn: Callable[[jax.Array], jax.Array] = jax.nn.selu,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0)
    ) -> None:
        super().__init__()
        
        self.act_fn = px.static(act_fn)
        
        # Create layers
        input_dim = 28 * 28  # Flattened MNIST image
        layer_dims = [input_dim] + hidden_dims + [10]  # 10 classes for MNIST
        
        # Split key for each layer
        keys = jax.random.split(key, len(layer_dims) - 1)
        self.layers = []
        for in_dim, out_dim, k in zip(layer_dims[:-1], layer_dims[1:], keys):
            layer = MyLinear(in_dim, out_dim, key=k)
            self.layers.append(layer)
            
        # Create VODEs
        self.vodes = [pxc.Vode() for _ in range(len(self.layers)-1)]
        self.vodes.append(pxc.Vode(pxc.ce_energy))  # Cross entropy for final layer
        self.vodes[-1].h.frozen = True

    def __call__(self, x, y):
        # Ensure input is properly shaped [batch_size, features]
        if x.ndim == 1:
            x = x[None, :]  # add batch dimension if single example
        else:
            x = x.reshape(x.shape[0], -1)  # Flatten while preserving batch dimension
        
        # Forward pass through hidden layers
        for v, l in zip(self.vodes[:-1], self.layers[:-1]):
            x = v(self.act_fn(l(x)))
            
        # Output layer
        x = self.vodes[-1](self.layers[-1](x))
        
        if y is not None:
            self.vodes[-1].set("h", y)
            
        return self.vodes[-1].get("u")

# Define forward and energy functions
@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: MNISTModel):
    return model(x, y)

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0, axis_name="batch")
def energy(x, y, *, model: MNISTModel):
    logits = model(x, None)
    # Compute softmax cross-entropy loss per example (summing over the class dimension)
    loss = optax.softmax_cross_entropy(logits, y).sum()
    return loss, logits

# Define a wrapper that sums over the batch to produce a scalar loss
def batch_energy(x, y, *, model: MNISTModel):
    losses, logits = energy(x, y, model=model)
    return jnp.sum(losses), logits

# Training function for one batch
@pxf.jit(static_argnums=0)
def train_on_batch(
    T: int,
    x: jax.Array,
    y: jax.Array,
    *,
    model: MNISTModel,
    optim_w: pxu.Optim,
    optim_h: pxu.Optim
):
    model.train()
    
    # Initialize
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    
    # Inference steps
    for i in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, logits), g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True
            )(batch_energy)(x, y, model=model)
        # Replace None gradients with zeros.
        g_fixed = jax.tree_util.tree_map(lambda x: 0.0 if x is None else x, g["model"])
        optim_h.step(model, g_fixed)
    
    # Weight update
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, logits), g = pxf.value_and_grad(
            pxu.M(pxnn.LayerParam).to([False, True]), 
            has_aux=True
        )(batch_energy)(x, y, model=model)
    # Replace None gradients with zeros.
    g_fixed = jax.tree_util.tree_map(lambda x: 0.0 if x is None else x, g["model"])
    optim_w.step(model, g_fixed, scale_by=1.0/x.shape[0])

# Evaluation function
@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: MNISTModel):
    model.eval()
    
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)
    
    return (y_ == y).mean()

# Main training loop
def main():
    # Hyperparameters
    batch_size = 128
    hidden_dims = [512, 256]  # Two hidden layers
    h_lr = 0.01      # Lower hidden learning rate
    w_lr = 0.001     # Weight learning rate remains unchanged
    num_epochs = 4   # Limit epochs to 4 as requested; improvement should be visible by end of epoch 1
    T = 20           # Inference steps remain the same
    
    # Initialize model with random key
    key = jax.random.PRNGKey(0)
    model = MNISTModel(hidden_dims, key=key)
    
    # Initialize optimizers
    optim_h = pxu.Optim(
        lambda: optax.chain(optax.clip(1.0), optax.adam(h_lr)), 
        pxu.M_hasnot(pxc.VodeParam, frozen=True)(model)
    )
    optim_w = pxu.Optim(
        lambda: optax.chain(optax.clip(1.0), optax.adam(w_lr)), 
        pxu.M(pxnn.LayerParam)(model)
    )
    
    # Get data
    train_batches, test_batches = get_datasets(batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        for x, y in train_batches:
            x = jnp.array(x)
            y = jax.nn.one_hot(jnp.array(y, dtype=jnp.int32), 10)
            train_on_batch(T, x, y, model=model, optim_w=optim_w, optim_h=optim_h)
        
        # Evaluation
        test_acc = []
        for x, y in test_batches:
            x = jnp.array(x)
            y = jnp.array(y, dtype=jnp.int32)
            acc = eval_on_batch(x, y, model=model)
            test_acc.append(acc)
        
        print(f"Epoch {epoch+1}, Test accuracy: {np.mean(test_acc):.4f}")

if __name__ == "__main__":
    main() 