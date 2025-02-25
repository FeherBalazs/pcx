import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax

import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.utils as pxu
import pcx.functional as pxf

from utils_dataloader import get_dataloaders

STATUS_FORWARD = "forward"
STATUS_REFINE = "refine"


import jax.random as jrandom
key = jrandom.PRNGKey(42)  # Same seed in both versions

class Decoder(pxc.EnergyModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nm_layers: int,
        act_fn: Callable[[jax.Array], jax.Array],
    ) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)

        self.layers = (
            [pxnn.Linear(input_dim, hidden_dim)]
            + [pxnn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)]
            + [pxnn.Linear(hidden_dim, output_dim)]
        )

        # We initialise the first node to zero.
        # We use 'zero_energy' as we do not want any prior on the first layer.
        self.vodes = (
            [
                pxc.Vode(
                    energy_fn=None,
                    ruleset={
                        pxc.STATUS.INIT: ("h, u <- u:to_zero",)},
                    tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros((input_dim,))},
                )
            ]
            + [
                # we stick with default forward initialisation for now for the remaining nodes,
                # however we enable a "forward mode" where we forward the incoming activation instead
                # of the node state; this is used during evaluation to generate the encoded output.
                pxc.Vode(
                    ruleset={
                        # pxc.STATUS.INIT: ("h, u <- u:to_zero",),
                        # STATUS_FORWARD: ("h -> u",),
                        # STATUS_REFINE: ("h <- u",)
                    },
                    tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros_like(v)},
                )
                for _ in range(nm_layers - 1)
            ]
            + [pxc.Vode()]
        )
        self.vodes[-1].h.frozen = True

    def __call__(self, y: jax.Array | None):
        x = self.vodes[0](jnp.empty(()))
        for i, layer in enumerate(self.layers):
            act_fn = self.act_fn if i != len(self.layers) - 1 else lambda x: x
            x = act_fn(layer(x))
            x = self.vodes[i + 1](x)

        if y is not None:
            self.vodes[-1].set("h", y.flatten())

        return self.vodes[-1].get("u")


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=0, out_axes=0)
def forward(x, *, model: Decoder):
    return model(x)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), out_axes=(None, 0), axis_name="batch")
def energy(*, model: Decoder):
    y_ = model(None)
    return jax.lax.psum(model.energy(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, *, model: Decoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    h_value, w_value, h_grad, w_grad = None, None, None, None

    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)

    learning_step = pxf.value_and_grad(pxu.M_hasnot(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)

    # Top down sweep and setting target value (do we need this? we could simply set the target value directly)
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)

    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference and learning steps
    # Here we could  add logic to do this until convergence for each sample or batch
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            h_value, h_grad = inference_step(model=model)
        optim_h.step(model, h_grad["model"])

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            w_value, w_grad = learning_step(model=model)
        optim_w.step(model, w_grad["model"], scale_by=1.0/x.shape[0])
    
    optim_h.clear()

    # with pxu.step(model):
    #     forward(x, model=model)

    return h_value, w_value, h_grad, w_grad


def train(dl, T, *, model: Decoder, optim_w: pxu.Optim, optim_h: pxu.Optim):

    for x, y in dl:
        # # Extract the first 4 pixel values from x
        # x_subset = x.numpy().flatten()[:4].reshape(1, 1, 2, 2)  # Reshape to maintain dimensions        
        # x = torch.tensor(x_subset)
        # print('x', x)

        h_value, w_value, h_grad, w_grad = train_on_batch(T, x.numpy(), model=model, optim_w=optim_w, optim_h=optim_h)

        # print('h_energy_sum: {}'.format(h_value[0]), '\nh_energy: {}'.format(h_value[1]))
        # print('w_energy_sum: {}'.format(w_value[0]), '\nw_energy: {}'.format(w_value[1]))
        # print('h_grad', h_grad['model'])
        # print('w_grad', w_grad['model'])    

        # for i, vode in enumerate(model.vodes):
        #         print('vode_h', i, vode.get('h'))
        # for i, vode in enumerate(model.vodes):
        #     try:
        #         print('vode_u', i, vode.get('u'))
        #     except:
        #         print('vode', i, vode)


@pxf.jit(static_argnums=0)
def eval_on_batch(T: int, x: jax.Array, *, model: Decoder, optim_h: pxu.Optim):
    model.eval()

    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(
        energy
    )

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)
    
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = inference_step(model=model)

        optim_h.step(model, g["model"])
    
    optim_h.clear()

    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat = forward(None, model=model)

    loss = jnp.square(jnp.clip(x_hat.flatten(), 0.0, 1.0) - x.flatten()).mean()

    return loss, x_hat


def eval(dl, T, *, model: Decoder, optim_h: pxu.Optim):
    losses = []

    for x, y in dl:
        e, y_hat = eval_on_batch(T, x.numpy(), model=model, optim_h=optim_h)
        losses.append(e)

    return np.mean(e)


def corrupt_input(x, corrupt_ratio=0.5):
    """Zero-out for specified portion"""
    corrupt_dim = int(x.shape[-1] * corrupt_ratio)
    corrupted = x.at[..., :corrupt_dim].set(0.)
    return corrupted


def eval_on_batch_for_vis(T: int, x: jax.Array, *, model: Decoder, optim_h: pxu.Optim, use_corruption: bool = False, corrupt_ratio: float = 0.5):
    """
    Runs inference on a batch (x) and returns the loss and reconstructed output (x_hat).
    If use_corruption is True, the first half (392 pixels) of each flattened image is set to black.
    Otherwise, the image is used unmodified.
    """
    model.eval()

    # Initialize a fresh optimizer state
    optim_h.clear()
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    # Use the regular energy function as set up for training
    inference_step = pxf.value_and_grad(
        pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
        has_aux=True
    )(energy)
    
    ################################################################
    # To ensure our caches are correctly initialized, we need to run a
    # forward pass on a batch matching the model's expected batch size.
    #
    # Determine the expected batch size. We'll look at the first VODE element.
    expected_bs = 1
    for vode in model.vodes:
        if vode.h._value is not None:
            expected_bs = vode.h._value.shape[0]
            break

    # If x's batch size does not match expected_bs, replicate along batch axis.
    if x.shape[0] != expected_bs:
        x_batch = jnp.repeat(x, expected_bs, axis=0)
    else:
        x_batch = x

    # Flatten x_batch to shape (batch_size, 784) since FashionMNIST images are 28x28.
    x_flat = jnp.reshape(x_batch, (x_batch.shape[0], -1))

    if use_corruption:
        x_input = corrupt_input(x_flat, corrupt_ratio=0.5)
    else:
        x_input = x_flat

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
    # with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        forward(x_input, model=model)   # Initialize caches with chosen input (corrupted or not)
    
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    # Inference iterations: update internal latent states.
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            h_value, h_grad = inference_step(model=model)
    
    optim_h.clear()
    
    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat_batch = forward(None, model=model)
    
    # x_batch is available from before; reshape it to compare with reconstructed images.
    x_orig_flat = jnp.reshape(x_batch, (x_batch.shape[0], -1))
    # Compute loss across the whole batch.
    loss = jnp.mean(jnp.square(jnp.clip(x_hat_batch, 0.0, 1.0) - x_orig_flat))
    return loss, x_hat_batch


def eval_on_batch_for_vis_partial(T: int, x: jax.Array, *, model: Decoder, optim_h: pxu.Optim, use_corruption: bool = False, corrupt_ratio: float = 0.5):
    """
    Runs inference on a batch (x) and returns the loss and reconstructed output (x_hat).
    If use_corruption is True, the first half (392 pixels) of each flattened image is set to black.
    Otherwise, the image is used unmodified.
    """
    model.eval()
    optim_h.clear()
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    # Use the regular energy function as set up for training
    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)
    
    # Get the model's expected batch size
    expected_bs = model.vodes[0].h._value.shape[0]
    
    # Repeat the input to match the expected batch size
    x = jnp.repeat(x, expected_bs, axis=0)

    # Flatten x_batch to shape (batch_size, 784) since FashionMNIST images are 28x28.
    x_flat = jnp.reshape(x, (x.shape[0], -1))

    # Create mask: True for known pixels (upper half), False for missing (lower half)
    mask = jnp.arange(784) < 784 * corrupt_ratio

    # Prepare input based on partial or full reconstruction
    if use_corruption:
        # Known pixels from x, missing pixels initialized to 0
        x_input = jnp.where(mask, x_flat, 0.0)
    else:
        x_input = x_flat

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x_input, model=model)
    
    # Inference iterations: update internal latent states.
    for _ in range(T):
        if use_corruption:
            # Unfreeze the last layer
            model.vodes[-1].h.frozen = False

            # Run inference step
            with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                h_value, h_grad = inference_step(model=model)

            # Zero gradients for known pixels in sensory layer
            sensory_h_grad = h_grad["model"].vodes[-1].h._value
            # print('sensory_h_grad', sensory_h_grad)

            # Ensure mask has compatible shape (e.g., [1, 784])
            mask_broadcasted = mask[None, :] if mask.ndim == 1 else mask

            # Apply the mask to the array
            modified_sensory_h_grad = jnp.where(mask_broadcasted, 0.0, sensory_h_grad)
            # print('modified_sensory_h_grad', modified_sensory_h_grad)

            # Update the VodeParam object with the modified array
            h_grad["model"].vodes[-1].h._value = modified_sensory_h_grad
        
        else:
            # Run inference step
            with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                h_value, h_grad = inference_step(model=model)
        
        # Update states with modified gradients
        optim_h.step(model, h_grad["model"])
    
    optim_h.clear()
    
    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat_batch = forward(None, model=model)
    
    # x_batch is available from before; reshape it to compare with reconstructed images.
    x_orig_flat = jnp.reshape(x, (x.shape[0], -1))

    # Compute loss across the whole batch.
    loss = jnp.mean(jnp.square(jnp.clip(x_hat_batch, 0.0, 1.0) - x_orig_flat))

    return loss, x_hat_batch


def visualize_reconstruction(model, optim_h, train_dataloader, T_values=[24], use_corruption=False, corrupt_ratio: float = 0.5, target_class: int = None):
    """
    Visualizes the reconstruction of images from the training DataLoader.
    If target_class is specified, only samples of that class are visualized.
    """
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    from datetime import datetime

    num_images = 5
    orig_images = []
    recon_images = {T: [] for T in T_values}  # Dictionary to store reconstructions for each T
    labels_list = []

    # Reset model state for visualization
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        x, _ = next(iter(train_dataloader))
        forward(x.numpy(), model=model)
    
    # Collect num_images samples from the DataLoader
    for i in range(num_images):
        x, label = next(iter(train_dataloader))  # Get a single image and its label
        x = jnp.array(x.numpy())

        # Get reconstructions for each T value
        for T in T_values:
            loss, x_hat = eval_on_batch_for_vis_partial(T, x, model=model, optim_h=optim_h, use_corruption=use_corruption, corrupt_ratio=corrupt_ratio)

            # Extract the reconstruction corresponding to the original sample
            x_hat_single = jnp.take(x_hat, 0, axis=0)
            recon_images[T].append(jnp.reshape(x_hat_single, (28, 28)))

        # Store original image and label
        orig_images.append(jnp.reshape(x[0, 0], (28, 28)))
        labels_list.append(label[0].item())

    # Create a grid plot with 5 rows and (1 + len(T_values)) columns
    fig, axes = plt.subplots(num_images, 1 + len(T_values), 
                            figsize=(4 * (1 + len(T_values)), 2 * num_images))
    
    for i in range(num_images):
        # Plot original image
        axes[i, 0].imshow(jnp.clip(orig_images[i], 0.0, 1.0), cmap='gray')
        axes[i, 0].set_title(f'Original (Label: {labels_list[i]})')
        axes[i, 0].axis('off')
        
        # Plot reconstructions for each T value
        for j, T in enumerate(T_values):
            axes[i, j+1].imshow(jnp.clip(recon_images[T][i], 0.0, 1.0), cmap='gray')
            axes[i, j+1].set_title(f'T={T}')
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    
    # Generate timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../results/reconstruction_{timestamp}.png"
    
    # Save the figure
    plt.savefig(filename)
    plt.close()

    # Return the list of original images and reconstructions
    return orig_images, recon_images


if __name__ == '__main__':
    batch_size = 16
    nm_epochs = 50
    target_class = None
    
    model = Decoder(
        input_dim=256, 
        hidden_dim=256, 
        output_dim=28 * 28,
        nm_layers=4, 
        act_fn=jax.nn.swish
    )
    
    optim_h = pxu.Optim(lambda: optax.sgd(5e-2, momentum=0.1))
    optim_w = pxu.Optim(lambda: optax.adamw(1e-4), pxu.M(pxnn.LayerParam)(model))

    # optim_h = pxu.Optim(lambda: optax.sgd(0.5, momentum=0.1))
    # optim_w = pxu.Optim(lambda: optax.adamw(0.00005), pxu.M(pxnn.LayerParam)(model))
    
    train_dataloader, test_dataloader = get_dataloaders(batch_size, train_subset_n=100, test_subset_n=100, target_class=target_class)
    
    # Init step setting up h and u value - Shall be done only at the beginning of the training
    x, _ = next(iter(train_dataloader))
    # x = torch.tensor(x.numpy().flatten()[:4].reshape(1, 1, 2, 2))
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x.numpy(), model=model)

    for e in range(nm_epochs):
        train(train_dataloader, T=8, model=model, optim_w=optim_w, optim_h=optim_h)
        l = eval(test_dataloader, T=8, model=model, optim_h=optim_h)
        print(f"Epoch {e + 1}/{nm_epochs} - Test Loss: {l:.4f}")
    
    x_orig, x_recon = visualize_reconstruction(model, optim_h, train_dataloader, T_values=[0, 1, 8, 64, 100], use_corruption=True, corrupt_ratio=0.5, target_class=target_class)

    # TODO: study mode collapse and add noise to the input