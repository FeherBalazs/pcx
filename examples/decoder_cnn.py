import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Callable
from functools import partial
from contextlib import contextmanager 

import jax
import jax.numpy as jnp
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
        latent_dim: int,              # Dimensionality of the top-level latent (e.g., 256)
        image_shape: tuple,           # Output image shape (channels, height, width)
        n_feat: int,                  # Number of feature maps in intermediate layers
        act_fn: Callable[[jax.Array], jax.Array],  # Activation function
        use_noise: bool = True,       # True: initialize latent as noise; False: zeros (for learned embeddings)
    ) -> None:
        super().__init__()
        self.image_shape = px.static(image_shape)
        self.output_dim = image_shape[0] * image_shape[1] * image_shape[2]
        self.latent_dim = latent_dim
        self.act_fn = px.static(act_fn)
        self.use_noise = use_noise


        # Initial spatial grid for convolutional processing
        self.h_init = 8
        self.w_init = 8
        self.channels_init = self.latent_dim // (self.h_init * self.w_init)
        if self.channels_init * self.h_init * self.w_init != self.latent_dim:
            raise ValueError("latent_dim must be divisible by h_init * w_init")

        # Reshape shape for convolutional layers (shape: (channels_init, h_init, w_init))
        self.reshape_shape = px.static((self.channels_init, self.h_init, self.w_init))

        # Create mask (same as in eval_on_batch_partial)
        channels, H, W = self.image_shape
        corrupt_ratio = 0.5  # You might want to make this configurable
        corrupt_height = int(corrupt_ratio * H)
        mask = jnp.ones((H, W), dtype=jnp.float32)
        mask = mask.at[corrupt_height:, :].set(0)
        mask_broadcasted = mask[None, None, :, :]  # Shape (1, 1, H, W)

        # Bind the mask to the energy function
        masked_energy = partial(masked_se_energy, mask=mask_broadcasted)

        # Define Vodes and collect them into a list
        self.vodes = [
            # Latent Vode (top-level representation)
            pxc.Vode(
                energy_fn=None,
                ruleset={pxc.STATUS.INIT: ("h, u <- u:to_init",)},
                # tforms={"to_init": lambda n, k, v, rkg: jnp.zeros((self.latent_dim))}
                tforms={"to_init": lambda n, k, v, rkg: jax.random.normal(px.RKG(), (self.latent_dim,)) * 0.01}
            ),
            # Vode after first upsampling layer
            pxc.Vode(
                ruleset={STATUS_FORWARD: ("h -> u",)}
                ),
            # Vode after second upsampling layer
            pxc.Vode(
                ruleset={STATUS_FORWARD: ("h -> u",)}
                ),
            # Vode after third upsampling layer
            pxc.Vode(
                ruleset={STATUS_FORWARD: ("h -> u",)}
                ),
            # Output Vode (sensory layer)
            pxc.Vode()
        ]
        # Freeze the output Vode's hidden state
        self.vodes[-1].h.frozen = True

        # Decoder layers: Convolutional transpose for upsampling
        self.up1 = pxnn.ConvTranspose(num_spatial_dims=2, in_channels=self.channels_init, out_channels=2 * n_feat, kernel_size=4, stride=2, padding=1)
        self.up2 = pxnn.ConvTranspose(2, 2 * n_feat, n_feat, kernel_size=4, stride=2, padding=1)
        self.refine = pxnn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)  # Refines features, keeps size
        self.out = pxnn.Conv2d(n_feat, 3, kernel_size=3, padding=1)


    def __call__(self, y: jax.Array | None = None):
        # Get the top-level latent from the first Vode
        x = self.vodes[0](jnp.empty(()))  # Shape: (latent_dim,)

        # Reshape for convolutional layers
        x = x.reshape(self.reshape_shape)  # Shape: (channels_init, h_init, w_init)

        # Upsampling path
        x = self.up1(x)
        x = self.act_fn(x)
        x = self.vodes[1](x)  # Vode after up1

        x = self.up2(x)
        x = self.act_fn(x)
        x = self.vodes[2](x)  # Vode after up2

        # Refinement 
        x = self.refine(x)  # (n_feat, 32, 32)
        x = self.act_fn(x)
        x = self.vodes[3](x)

        # Output layer (sensory layer)
        x = self.out(x)
        x = self.vodes[4](x)  # Output Vode, Shape: (channels, height, width)

        if y is not None:
            # If target image is provided, set the sensory Vode's hidden state
            self.vodes[4].set("h", y)  # y should be (channels, height, width)

        return self.vodes[4].get("u")  # Return the predicted image


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=0, out_axes=0)
def forward(x, *, model: Decoder):
    return model(x)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), out_axes=(None, 0), axis_name="batch")
def energy(*, model: Decoder):
    y_ = model(None)
    return jax.lax.psum(model.energy(), "batch"), y_


@contextmanager
def temp_set_energy_fn(vode, new_energy_fn):
    """Temporarily set a Vode's energy_fn to a px.static-wrapped function."""
    original_energy_fn = vode.energy_fn  # Save the original
    vode.energy_fn = px.static(new_energy_fn)  # Set the new one
    try:
        yield
    finally:
        vode.energy_fn = original_energy_fn  # Restore the original


def masked_se_energy(vode, rkg, mask):
    """
    Compute the masked sensory energy, considering only known pixels.
    
    Args:
        vode: The Vode object containing h and u.
        rkg: Unused (required for energy function signature).
        mask: The mask array, where 1 indicates known pixels and 0 indicates masked pixels.
              Shape should be compatible with h and u (e.g., (1, 1, H, W)).
    
    Returns:
        The masked energy (scalar).
    """
    h = vode.get("h")  # Shape (batch_size, channels, H, W)
    u = vode.get("u")  # Shape (batch_size, channels, H, W)
    error = (h - u) ** 2  # Shape (batch_size, channels, H, W)
    masked_error = error * mask  # Zero out error for masked pixels
    return jnp.sum(masked_error) / jnp.sum(mask)  # Normalize by number of known pixels


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, *, model: Decoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    h_energy, w_energy, h_grad, w_grad = None, None, None, None

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
            (h_energy, y_), h_grad = inference_step(model=model)
        optim_h.step(model, h_grad["model"])

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (w_energy, y_), w_grad = learning_step(model=model)
        optim_w.step(model, w_grad["model"], scale_by=1.0/x.shape[0])
    
    optim_h.clear()

    return h_energy, w_energy, h_grad, w_grad


def train(dl, T, *, model: Decoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for x, y in dl:
        h_energy, w_energy, h_grad, w_grad = train_on_batch(T, x.numpy(), model=model, optim_w=optim_w, optim_h=optim_h)


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
            (h_energy, y_), h_grad = inference_step(model=model)

        optim_h.step(model, h_grad["model"])
    
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


# @pxf.jit(static_argnums=(0, 1))
def eval_on_batch_partial(use_corruption: bool, corrupt_ratio: float, T: int, x: jax.Array, *, model: Decoder, optim_h: pxu.Optim):
    """
    Runs inference on a batch (x) and returns the reconstructed output (x_hat).
    If use_corruption is True, applies a mask to the input and uses masked energy.
    """
    model.eval()
    optim_h.clear()
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Define inference step with the regular energy function
    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)

    # Determine expected batch size from model state
    expected_bs = 1
    for vode in model.vodes:
        if vode.h._value is not None:
            expected_bs = vode.h._value.shape[0]
            break

    # Adjust batch size if needed
    if x.shape[0] != expected_bs:
        x_batch = jnp.repeat(x, expected_bs, axis=0)
    else:
        x_batch = x

    batch_size, channels, H, W = x_batch.shape
    assert model.image_shape.get() == (channels, H, W), "Image shape mismatch"

    # TODO: Masking is not sure if working as expected.
    # Prepare input and energy function based on corruption
    if use_corruption:
        # Create spatial mask
        corrupt_height = int(corrupt_ratio * H)  # e.g., 16 for H=32, ratio=0.5
        mask = jnp.ones((H, W), dtype=jnp.float32)
        mask = mask.at[corrupt_height:, :].set(0)  # Mask bottom portion
        mask_broadcasted = mask[None, None, :, :]  # Shape: (1, 1, H, W)
        x_input = x_batch * mask_broadcasted  # Corrupt the input
        # Define masked energy function
        energy_fn_to_use = px.static(lambda vode, rkg: masked_se_energy(vode, rkg, mask=mask_broadcasted))
    else:
        x_input = x_batch
        energy_fn_to_use = px.static(pxc.se_energy)  # Use standard energy

    # Initialize the model with the input
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x_batch, model=model)

    # Temporarily set the energy function
    original_energy_fn = model.vodes[-1].energy_fn
    model.vodes[-1].energy_fn = energy_fn_to_use

    # Temporarily set the energy function using the context manager
    with temp_set_energy_fn(model.vodes[-1], energy_fn_to_use):
        # Inference iterations
        for _ in range(T):
            if use_corruption:
                # Unfreeze sensory layer for inference
                model.vodes[-1].h.frozen = False

                # Run inference step
                with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                    h_energy, h_grad = inference_step(model=model)

                    # Zero gradients for known (unmasked) pixels
                    sensory_h_grad = h_grad["model"].vodes[-1].h._value
                    modified_sensory_h_grad = jnp.where(mask_broadcasted, 0.0, sensory_h_grad)
                    h_grad["model"].vodes[-1].h._value = modified_sensory_h_grad
            else:
                # Standard inference step
                with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                    h_energy, h_grad = inference_step(model=model)

            # Update states
            optim_h.step(model, h_grad["model"])

    optim_h.clear()

    # Final forward pass to get reconstruction
    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat_batch = forward(None, model=model)

    return x_hat_batch


def visualize_reconstruction(model, optim_h, dataloader, T_values=[24], use_corruption=False, corrupt_ratio=0.5, target_class: int = None, num_images: int = 2):
    import matplotlib.pyplot as plt
    from datetime import datetime

    orig_images = []
    recon_images = {T: [] for T in T_values}
    labels_list = []
    dataloader_iter = iter(dataloader)
    
    # Load and reshape images
    for _ in range(num_images):
        x, label = next(dataloader_iter)
        x = jnp.array(x.numpy())
        
        orig_images.append(jnp.reshape(x[0], model.image_shape.get()))
        
        # Handle label as scalar or 1-element array
        labels_list.append(label[0].item() if len(label.shape) > 0 else label.item())
        for T in T_values:
            x_hat = eval_on_batch_partial(use_corruption=use_corruption, corrupt_ratio=corrupt_ratio, T=T, x=x, model=model, optim_h=optim_h)
            x_hat_single = jnp.reshape(x_hat[0], model.image_shape.get())
            recon_images[T].append(x_hat_single)
    
    # Create subplots
    fig, axes = plt.subplots(num_images, 1 + len(T_values), figsize=(4 * (1 + len(T_values)), 2 * num_images))
    
    # If num_images = 1, make axes 2D by adding a row dimension
    if num_images == 1:
        axes = axes[None, :]  # Shape becomes (1, 1 + len(T_values))
    
    # Plot images
    for i in range(num_images):
        # Check number of channels with .get()
        if model.image_shape.get()[0] == 1:  # Grayscale
            axes[i, 0].imshow(jnp.clip(jnp.squeeze(orig_images[i]), 0.0, 1.0), cmap='gray')
        else:  # RGB
            axes[i, 0].imshow(jnp.clip(jnp.transpose(orig_images[i], (1, 2, 0)), 0.0, 1.0))
        axes[i, 0].set_title(f'Original (Label: {labels_list[i]})')
        axes[i, 0].axis('off')
        
        for j, T in enumerate(T_values):
            if model.image_shape.get()[0] == 1:  # Grayscale
                axes[i, j+1].imshow(jnp.clip(jnp.squeeze(recon_images[T][i]), 0.0, 1.0), cmap='gray')
            else:  # RGB
                axes[i, j+1].imshow(jnp.clip(jnp.transpose(recon_images[T][i], (1, 2, 0)), 0.0, 1.0))
            axes[i, j+1].set_title(f'T={T}')
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"../results/reconstruction_{timestamp}.png")
    plt.close()
    return orig_images, recon_images


if __name__ == '__main__':
    # Added dataset selection and configuration
    dataset_name = "cifar10"  # Change to "cifar10" or "imagenet" "fashionmnist" as needed
    root_path = "../datasets/"  # Adjust to your dataset root path
    batch_size = 100
    nm_epochs = 20
    target_class = None
    num_images = 5
    train_subset_n = 1000
    test_subset_n = 1000
    
    latent_dim = 4096
    n_feat = 128
    
    # Define image_shape and output_dim based on dataset
    if dataset_name == "fashionmnist":
        image_shape = (1, 28, 28)
        output_dim = 28 * 28
    elif dataset_name == "cifar10":
        image_shape = (3, 32, 32)
        output_dim = 32 * 32 * 3
    elif dataset_name == "imagenet":
        image_shape = (3, 224, 224)
        output_dim = 224 * 224 * 3
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    

    model = Decoder(
        latent_dim=latent_dim,
        image_shape=image_shape,
        n_feat=n_feat,
        act_fn=jax.nn.swish,
        use_noise=False,  # Start with noise; set to False for learned embeddings
    )
    
    optim_h = pxu.Optim(lambda: optax.sgd(5e-1, momentum=0.1))
    # optim_h = pxu.Optim(lambda: optax.sgd(5e-2, momentum=0.2))
    optim_w = pxu.Optim(lambda: optax.adamw(1e-4), pxu.M(pxnn.LayerParam)(model))
    
    # Updated get_dataloaders call to include dataset_name and root_path
    train_dataloader, test_dataloader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        root_path=root_path,
        train_subset_n=train_subset_n,
        test_subset_n=test_subset_n,
        target_class=target_class
    )
    
    x, _ = next(iter(train_dataloader))
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x.numpy(), model=model)
    
    for e in range(nm_epochs):
        train(train_dataloader, T=8, model=model, optim_w=optim_w, optim_h=optim_h)
        l = eval(test_dataloader, T=8, model=model, optim_h=optim_h)
        print(f"Epoch {e + 1}/{nm_epochs} - Test Loss: {l:.4f}")
    
    visualize_reconstruction(model, optim_h, train_dataloader, T_values=[0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,32,64], use_corruption=False, corrupt_ratio=0.25, target_class=target_class, num_images=num_images)