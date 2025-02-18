from typing import Callable

import jax
import jax.numpy as jnp

import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.utils as pxu
import equinox as eqx

STATUS_FORWARD = "forward"


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

        self.input_dim = input_dim  # store input dimension for later use
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
                    ruleset={pxc.STATUS.INIT: ("h, u <- u:to_zero",)},
                    tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros((input_dim,))},
                )
            ]
            + [
                # we stick with default forward initialisation for now for the remaining nodes,
                # however we enable a "forward mode" where we forward the incoming activation instead
                # of the node state; this is used during evaluation to generate the encoded output.
                pxc.Vode(
                    ruleset={
                        pxc.STATUS.INIT: ("h, u <- u:to_zero",),
                        STATUS_FORWARD: ("h -> u",)
                    },
                    tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros_like(v)},
                )
                for _ in range(nm_layers - 1)
            ]
            + [pxc.Vode()]
        )
        self.vodes[-1].h.frozen = True

    def __call__(self, y: jax.Array | None):
        # The defined ruleset for the first vode is to set the hidden state to zero,
        # independent of the input, so we always pass '-1' (as None would skip the computation).
        x = self.vodes[0](jnp.empty(()))
        for i, layer in enumerate(self.layers):
            act_fn = self.act_fn if i != len(self.layers) - 1 else lambda x: x
            x = act_fn(layer(x))
            x = self.vodes[i + 1](x)

        if y is not None:
            self.vodes[-1].set("h", y.flatten())

        return self.vodes[-1].get("u")

import torch
import numpy as np


# The dataloader assumes cuda is being used, as such it sets 'pin_memory = True' and
# 'prefetch_factor = 2'. Note that the batch size should be constant during training, so
# we set 'drop_last = True' to avoid having to deal with variable batch sizes.
class TorchDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=1,
        pin_memory=True,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=True,
        prefetch_factor=2,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True if batch_sampler is None else None,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

import torchvision
import torchvision.transforms as transforms


def get_dataloaders(batch_size: int, train_subset_n: int = None, test_subset_n: int = None, target_class: int = None):
    t = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    train_dataset = torchvision.datasets.FashionMNIST(
        "~/tmp/fashion-mnist/",
        transform=t,
        download=True,
        train=True,
    )
    from torch.utils.data import Subset
    # If target_class is specified, filter the dataset to only include that category.
    if target_class is not None:
        # Obtain indices where the target equals the target_class. FashionMNIST stores targets as a tensor.
        target_indices = (train_dataset.targets == target_class).nonzero(as_tuple=True)[0].tolist()
        train_dataset = Subset(train_dataset, target_indices)

    # Optionally restrict the training dataset further.
    if train_subset_n is not None:
        all_idx = list(range(len(train_dataset)))
        train_dataset = Subset(train_dataset, all_idx[:train_subset_n])

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        "~/tmp/fashion-mnist/",
        transform=t,
        download=True,
        train=False,
    )
    # If target_class is specified, filter to only that category.
    if target_class is not None:
        target_indices = (test_dataset.targets == target_class).nonzero(as_tuple=True)[0].tolist()
        test_dataset = Subset(test_dataset, target_indices)

    # Similarly, restrict the test dataset if required.
    if test_subset_n is not None:
        all_idx = list(range(len(test_dataset)))
        test_dataset = Subset(test_dataset, all_idx[:test_subset_n])

    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_dataloader, test_dataloader

import pcx.functional as pxf


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

    inference_step = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(
        energy
    )

    learning_step = pxf.value_and_grad(pxu.M_hasnot(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)

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

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = learning_step(model=model)
    optim_w.step(model, g["model"], scale_by=1.0/x.shape[0])


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


def train(dl, T, *, model: Decoder, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for x, y in dl:
        train_on_batch(T, x.numpy(), model=model, optim_w=optim_w, optim_h=optim_h)


def eval(dl, T, *, model: Decoder, optim_h: pxu.Optim):
    losses = []

    for x, y in dl:
        e, y_hat = eval_on_batch(T, x.numpy(), model=model, optim_h=optim_h)
        losses.append(e)

    return np.mean(e)


def partial_stop_gradient(x, pixel_dim=392):
    """
    Applies stop_gradient to the pixel part of x (first `pixel_dim` columns)
    while leaving the label part unaffected.
    
    Args:
        x (jax.Array): Concatenated tensor of pixel values and labels.
        pixel_dim (int): Number of dimensions for pixels (default 784).
        
    Returns:
        jax.Array: A tensor where gradients do not flow through the pixel portion.
    """
    # Freeze pixel part:
    pixels = jax.lax.stop_gradient(x[:, :pixel_dim])
    # Keep label part active:
    labels = x[:, pixel_dim:]
    # Concatenate them back
    return jnp.concatenate([pixels, labels], axis=-1)


def eval_on_batch_for_vis(T: int, x: jax.Array, *, model: Decoder, optim_h: pxu.Optim, use_corruption: bool = False):
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
        # Corrupt the first half (392 elements) by setting them to black.
        n_pixels = x_flat.shape[1]   # expected to be 784 for FashionMNIST
        half = n_pixels // 2
        x_input = jnp.concatenate([jnp.zeros((x_flat.shape[0], half)), x_flat[:, half:]], axis=1)
        x_input = partial_stop_gradient(x_input, half)
    else:
        # Use the unmodified flattened image.
        x_input = x_flat

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x_input, model=model)   # Initialize caches with chosen input (corrupted or not)
    
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    # Inference iterations: update internal latent states.
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = inference_step(model=model)
        optim_h.step(model, g["model"])
    
    optim_h.clear()
    
    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat_batch = forward(None, model=model)
    
    # x_batch is available from before; reshape it to compare with reconstructed images.
    x_orig_flat = jnp.reshape(x_batch, (x_batch.shape[0], -1))
    # Compute loss across the whole batch.
    loss = jnp.mean(jnp.square(jnp.clip(x_hat_batch, 0.0, 1.0) - x_orig_flat))
    return loss, x_hat_batch


def visualize_reconstruction(model, optim_h, T=24, dataset='test', use_corruption=False, target_class: int = None):
    """
    Loads one sample from FashionMNIST and shows a side-by-side plot of
    the original image and its reconstruction.
    """
    import matplotlib.pyplot as plt
    import torchvision
    import torchvision.transforms as transforms
    import jax.numpy as jnp
    import torch

    t = transforms.Compose([transforms.ToTensor()])
    ds = torchvision.datasets.FashionMNIST(
        "~/tmp/fashion-mnist/",
        transform=t,
        download=True,
        train=(dataset=='train')
    )
    from torch.utils.data import Subset
    if target_class is not None:
        # Filter the dataset so that only samples of the target class are kept.
        target_indices = (ds.targets == target_class).nonzero(as_tuple=True)[0].tolist()
        ds = Subset(ds, target_indices)
    else:
        ds = Subset(ds, list(range(100)))

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    it = iter(loader)

    num_images = 5
    orig_images = []
    recon_images = []
    labels_list = []
    for i in range(num_images):
        x, label = next(it)  # x shape: (1, 1, 28, 28)
        x = jnp.array(x.numpy())
        loss, x_hat = eval_on_batch_for_vis(T, x, model=model, optim_h=optim_h, use_corruption=use_corruption)
        # Since the model's internal state expects batch size 100,
        # the input of batch size 1 is replicated to produce x_hat with shape (100, 784).
        # Extract the reconstruction corresponding to the original sample.
        x_hat_single = jnp.take(x_hat, 0, axis=0)
        # x is shape (1, 1, 28, 28) -> reshape to (28,28)
        orig_images.append(jnp.reshape(x[0, 0], (28, 28)))
        # Reshape the single reconstruction to (28,28)
        recon_images.append(jnp.reshape(x_hat_single, (28, 28)))
        labels_list.append(label.item())
    
    # Create a grid plot with 5 rows and 2 columns (Original vs Reconstruction)
    fig, axes = plt.subplots(num_images, 2, figsize=(8, 2 * num_images))
    for i in range(num_images):
        axes[i, 0].imshow(jnp.clip(orig_images[i], 0.0, 1.0), cmap='gray')
        axes[i, 0].set_title(f'Original (Label: {labels_list[i]})')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(jnp.clip(recon_images[i], 0.0, 1.0), cmap='gray')
        axes[i, 1].set_title('Reconstruction')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()

    # Return the list of original images and reconstructions
    return orig_images, recon_images

import optax

# Global dictionaries for logging activations and weights over epochs.
activation_log = {}
weight_log = {}

def forward_with_logging(x, model: Decoder):
    """
    A wrapper for the decoder's forward pass that logs intermediate activations.
    
    We follow the same ordering as in the original __call__:
      - The first VODE module is applied.
      - Then, for every layer, we record:
          • The linear output of the layer.
          • The output after applying the activation function.
          • The subsequent VODE output.
      
    Finally, we record the network output.
    
    Returns:
        activations (dict): A dictionary with keys for each stage (e.g. "linear_0", "act_0", etc.).
        final_out: The final output from the network (i.e. the reconstruction).
    """
    activations = {}
    # Initialize the first VODE module exactly as in __call__.
    current = model.vodes[0](jnp.empty(()))
    activations["vode0"] = {"u": model.vodes[0].get("u"), "h": model.vodes[0].get("h")}

    for i, layer in enumerate(model.layers):
        # Compute the raw linear output.
        linear_out = layer(current)
        activations[f"linear_{i}"] = linear_out
        
        # Apply activation (if not the last layer, use model.act_fn, else identity)
        act_fn = model.act_fn if i != len(model.layers) - 1 else (lambda x: x)
        activated = act_fn(linear_out)
        activations[f"act_{i}"] = activated
        
        # Pass through the corresponding VODE module and update current with its output.
        current = model.vodes[i + 1](activated)
        activations[f"vode_{i+1}"] = {"u": model.vodes[i+1].get("u"), "h": model.vodes[i+1].get("h")}

    # Get the final output from the last VODE.
    final_u = model.vodes[-1].get("u")
    final_h = model.vodes[-1].get("h")
    activations["output"] = {"u": final_u, "h": final_h}
    final_out = final_u
    return activations, final_out

def log_weights(model: Decoder):
    """
    Log the weights for each layer in the decoder.
    
    We assume that each layer in model.layers is a Linear layer that has attributes
    'W' (weights) and 'b' (bias). Modify accordingly if your implementation differs.
    
    Returns:
        weights (dict): A dictionary keyed by layer indices containing a dict with keys "W" and "b".
    """
    weights = {}
    for i, layer in enumerate(model.layers):
        try:
            # For example, if the layer was created via pxnn.Linear it might store weights as:
            w = layer.W
            b = layer.b
        except AttributeError:
            # Fallback: If weights are stored in a 'params' dictionary.
            params = getattr(layer, "params", {})
            w = params.get("W", None)
            b = params.get("b", None)
        w_val = jnp.array(w) if w is not None else None
        b_val = jnp.array(b) if b is not None else None
        weights[f"layer_{i}"] = {"W": w_val, "b": b_val}
    return weights

def update_logs(epoch, activations, weights):
    """
    Save the activations and weights for the given epoch.
    """
    activation_log[epoch] = activations
    weight_log[epoch] = weights

def plot_activations_over_time(layer_key="act_0"):
    """
    Produce a heatmap that shows how the activations in a chosen layer evolve over epochs.
    If the activation tensor has a batch dimension, we average over it (yielding Epoch x Neuron).
    """
    epochs = sorted(activation_log.keys())
    act_evolution = []
    for ep in epochs:
        act = activation_log[ep].get(layer_key)
        if act is None:
            continue
        # Average over batch dimension if needed.
        act_mean = jnp.mean(act, axis=0) if act.ndim > 1 else act
        act_evolution.append(np.array(act_mean))
    act_evolution = np.stack(act_evolution, axis=0)
    plt.figure(figsize=(8,6))
    plt.imshow(act_evolution, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(f"Evolution of Activations for {layer_key}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Epoch")
    plt.show()

def plot_weight_histogram(layer_key="layer_0", param="W"):
    """
    Plot a histogram of weight values from a selected layer accumulated over epochs.
    """
    epochs = sorted(weight_log.keys())
    weight_values = []
    for ep in epochs:
        w = weight_log[ep].get(layer_key, {}).get(param)
        if w is not None:
            weight_values.extend(np.ravel(np.array(w)))
    plt.figure(figsize=(8,6))
    plt.hist(weight_values, bins=50)
    plt.title(f"Histogram of {param} in {layer_key} (over epochs)")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.show()

# An optional helper for stopping gradient flow on part of a tensor (if needed)
def partial_stop_gradient(x, pixel_dim=392):
    """
    Stops gradients for the first `pixel_dim` elements of x.
    """
    pixels = jax.lax.stop_gradient(x[:, :pixel_dim])
    labels = x[:, pixel_dim:]
    return jnp.concatenate([pixels, labels], axis=-1)

def main():
    # For demonstration, we log a fixed sample over a few epochs.
    batch_size = 1
    nm_epochs = 2
    
    # Load dataloaders. Adjust target_class or subset counts as needed.
    train_dl, test_dl = get_dataloaders(batch_size, train_subset_n=100, test_subset_n=100)
    
    # Create a model instance.
    model = Decoder(
        input_dim=64,
        hidden_dim=512,
        output_dim=28 * 28,
        nm_layers=2,
        act_fn=jax.nn.swish
    )
    
    # Create simple optimizers (only for demonstration; we do not perform full updates here).
    optim_h = optax.sgd(0.05, momentum=0.1)
    optim_w = optax.adamw(1e-4)
    
    # For visual logging, we use a fixed sample from the test set.
    import torch
    it = iter(test_dl)
    x, label = next(it)  # x shape: (1, 1, 28, 28)
    x = jnp.array(x.numpy())
    
    for epoch in range(nm_epochs):
        # Instead of training updates, we run a forward pass with logging.
        activations, out = forward_with_logging(x, model)
        weights = log_weights(model)
        update_logs(epoch, activations, weights)
        print(f"Logged activations and weights for epoch {epoch}")
    
    # Visualize logged activations and weights.
    plot_activations_over_time("act_0")
    plot_weight_histogram("layer_0", "W")
    
if __name__ == "__main__":
    main()