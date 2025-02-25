import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Callable
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
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nm_layers: int,
        act_fn: Callable[[jax.Array], jax.Array],
        image_shape: tuple,  # New parameter for image shape (channels, height, width)
    ) -> None:
        super().__init__()
        self.image_shape = image_shape  # Store image shape
        self.output_dim = output_dim    # Store output dimension
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
                        STATUS_FORWARD: ("h -> u",),
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

    return h_value, w_value, h_grad, w_grad


def train(dl, T, *, model: Decoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for x, y in dl:
        h_value, w_value, h_grad, w_grad = train_on_batch(T, x.numpy(), model=model, optim_w=optim_w, optim_h=optim_h)


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


def eval_on_batch_partial(T: int, x: jax.Array, *, model: Decoder, optim_h: pxu.Optim, use_corruption: bool = False, corrupt_ratio: float = 0.5):
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

    # Flatten x_batch to shape (batch_size, output_dim) for flexibility with different image sizes
    x_flat = jnp.reshape(x_batch, (x_batch.shape[0], -1))

    # Create mask: True for known pixels (first corrupt_ratio portion), False for missing
    mask = jnp.arange(model.output_dim) < model.output_dim * corrupt_ratio

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

            # Ensure mask has compatible shape (e.g., [1, output_dim])
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
    x_orig_flat = jnp.reshape(x_batch, (x_batch.shape[0], -1))

    # Compute loss across the whole batch.
    loss = jnp.mean(jnp.square(jnp.clip(x_hat_batch, 0.0, 1.0) - x_orig_flat))

    return loss, x_hat_batch


def visualize_reconstruction(model, optim_h, dataloader, T_values=[24], use_corruption=False, corrupt_ratio=0.5, target_class: int = None):
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    num_images = 5
    orig_images = []
    recon_images = {T: [] for T in T_values}
    labels_list = []
    dataloader_iter = iter(dataloader)
    
    for _ in range(num_images):
        x, label = next(dataloader_iter)
        x = jnp.array(x.numpy())
        # Use model.image_shape for reshaping to support different datasets
        orig_images.append(jnp.reshape(x[0], model.image_shape))
        labels_list.append(label[0].item())
        for T in T_values:
            _, x_hat = eval_on_batch_partial(T, x, model=model, optim_h=optim_h, use_corruption=use_corruption, corrupt_ratio=corrupt_ratio)
            # Reshape x_hat using model.image_shape
            x_hat_single = jnp.reshape(x_hat[0], model.image_shape)
            recon_images[T].append(x_hat_single)
    
    fig, axes = plt.subplots(num_images, 1 + len(T_values), figsize=(4 * (1 + len(T_values)), 2 * num_images))
    for i in range(num_images):
        # Handle both grayscale and RGB images based on image_shape
        if model.image_shape[0] == 1:  # Grayscale
            axes[i, 0].imshow(jnp.clip(jnp.squeeze(orig_images[i]), 0.0, 1.0), cmap='gray')
        else:  # RGB
            axes[i, 0].imshow(jnp.clip(jnp.transpose(orig_images[i], (1, 2, 0)), 0.0, 1.0))
        axes[i, 0].set_title(f'Original (Label: {labels_list[i]})')
        axes[i, 0].axis('off')
        for j, T in enumerate(T_values):
            if model.image_shape[0] == 1:  # Grayscale
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
    batch_size = 16
    nm_epochs = 25
    target_class = None
    
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
        input_dim=64, 
        hidden_dim=256, 
        output_dim=output_dim,  # Use dynamic output_dim
        nm_layers=3, 
        act_fn=jax.nn.swish,
        image_shape=image_shape  # Pass image_shape parameter
    )
    
    optim_h = pxu.Optim(lambda: optax.sgd(5e-2, momentum=0.1))
    optim_w = pxu.Optim(lambda: optax.adamw(1e-4), pxu.M(pxnn.LayerParam)(model))
    
    # Updated get_dataloaders call to include dataset_name and root_path
    train_dataloader, test_dataloader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        root_path=root_path,
        train_subset_n=100,
        test_subset_n=100,
        target_class=target_class
    )
    
    x, _ = next(iter(train_dataloader))
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x.numpy(), model=model)
    
    for e in range(nm_epochs):
        train(train_dataloader, T=64, model=model, optim_w=optim_w, optim_h=optim_h)
        l = eval(test_dataloader, T=64, model=model, optim_h=optim_h)
        print(f"Epoch {e + 1}/{nm_epochs} - Test Loss: {l:.4f}")
    
    visualize_reconstruction(model, optim_h, train_dataloader, T_values=[0, 1, 8, 64, 128, 256], use_corruption=True, corrupt_ratio=0.5, target_class=target_class)