from typing import Callable

import jax
import jax.numpy as jnp
import jax.lax

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

    # def inference(self, y: jax.Array | None):
    #     x = self.vodes[0](jnp.empty(()))
    #     for i, layer in enumerate(self.layers):
    #         act_fn = self.act_fn if i != len(self.layers) - 1 else lambda x: x
    #         x = act_fn(layer(x))
    #         x = self.vodes[i + 1](x)

    #     if y is not None:
    #       # Assuming full target vector is [pixels, label] and we want to only clamp the label part
    #       current_h = self.vodes[-1].get("h")
    #       image_dim = 784   # or whichever is appropriate in your case
    #       new_h = jnp.concatenate([current_h[:image_dim], y.flatten()], axis=0)
    #       self.vodes[-1].set("h", new_h)

    #     return self.vodes[-1].get("u")

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


def get_dataloaders(batch_size: int, train_subset_n: int = None, test_subset_n: int = None):
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
    # If a subset size is provided, restrict the training dataset.
    if train_subset_n is not None:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, list(range(train_subset_n)))

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
    # Similarly, restrict the test dataset if required.
    if test_subset_n is not None:
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, list(range(test_subset_n)))

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


# @pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=0, out_axes=0)
# def inference(x, *, model: Decoder):
#     return model.inference(x)


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)),
          out_axes=(None, 0),
          axis_name="batch")
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
    import numpy as np
    for x, y in dl:
        # Flatten image: (batch, 1, 28, 28) -> (batch, 784)
        x_flat = x.numpy().reshape(x.shape[0], -1)
        # One-hot encode labels (assume 10 classes)
        y_onehot = np.eye(10)[y.numpy().flatten()]
        # Concatenate image and label -> (batch, 794)
        input_vec = np.concatenate([x_flat, y_onehot], axis=1)
        train_on_batch(T, input_vec, model=model, optim_w=optim_w, optim_h=optim_h)


def eval(dl, T, *, model: Decoder, optim_h: pxu.Optim):
    import numpy as np
    losses = []
    for x, y in dl:
        x_flat = x.numpy().reshape(x.shape[0], -1)
        y_onehot = np.eye(10)[y.numpy().flatten()]
        input_vec = np.concatenate([x_flat, y_onehot], axis=1)
        e, y_hat = eval_on_batch(T, input_vec, model=model, optim_h=optim_h)
        losses.append(e)
    return np.mean(losses)


def partial_stop_gradient(x, pixel_dim=784):
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


def eval_on_batch_for_vis(T: int, x: jax.Array, *, model: Decoder, optim_h: pxu.Optim):
    """
    Runs inference on a batch (x) and returns loss and reconstructed output (x_hat).

    In this version, we assume that `x` is a concatenation of pixels (first 784 elements)
    and labels (next 10 elements). We use partial_stop_gradient to prevent pixel values from
    contributing to the gradients during inference.
    """
    model.eval()

    # Initialize a fresh optimizer state
    optim_h.clear()
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    # Determine the expected batch size from one of the caches.
    expected_bs = 1
    for vode in model.vodes:
        if vode.h._value is not None:
            expected_bs = vode.h._value.shape[0]
            break

    # Ensure our input batch has the right batch dimension:
    if x.shape[0] != expected_bs:
        x_batch = jnp.repeat(x, expected_bs, axis=0)
    else:
        x_batch = x
    # Now, replace the pixel part (first 784 entries) with all-black pixels.
    x_black = jnp.concatenate([jnp.zeros((x_batch.shape[0], 784)), x_batch[:, 784:]], axis=1)
    
    # Create the inference step using the masked value_and_grad.
    def energy_with_partial_stop(x, model):
        return energy(partial_stop_gradient(x), model=model)

    inference_step = pxf.value_and_grad(
        pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
        has_aux=True
    )(energy_with_partial_stop)
    
    ################################################################
    # Use a forward pass with the partially frozen all-black image input to initialize caches.
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        # forward(partial_stop_gradient(x_black), model=model)
        forward(x_black, model=model)
    ################################################################
    
    # Reinitialize the optimizer state for latent updates
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
    # Inference iterations: update internal latent states.
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = inference_step(x_batch, model=model)
        optim_h.step(model, g["model"])
    
    optim_h.clear()
    
    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat_batch = forward(None, model=model)
    
    # Extract the reconstruction for the first sample of the batch.
    x_hat = jnp.take(x_hat_batch, 0, axis=0)
    loss = jnp.square(jnp.clip(x_hat.flatten(), 0.0, 1.0) - x[0].flatten()).mean()
    return loss, x_hat


def visualize_reconstruction(model, optim_h, T=24, dataset='test'):
    """
    Loads one sample from FashionMNIST and shows a side-by-side plot of
    the original image and its reconstruction.
    """
    import matplotlib.pyplot as plt
    import torchvision
    import torchvision.transforms as transforms
    import jax.numpy as jnp
    import torch

    # Fetch one sample.
    t = transforms.Compose([transforms.ToTensor()])
    ds = torchvision.datasets.FashionMNIST(
        "~/tmp/fashion-mnist/",
        transform=t,
        download=True,
        train=(dataset=='train')
    )
    # Restrict the dataset to only 100 samples.
    from torch.utils.data import Subset
    ds = Subset(ds, list(range(100)))
    
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    x, label = next(iter(loader))
    
    # Process image and label: flatten image and one-hot encode label, then concatenate.
    import numpy as np
    x_flat = x.numpy().reshape(1, -1)
    y_onehot = np.eye(10)[label.numpy().flatten()]
    input_vec = np.concatenate([x_flat, y_onehot], axis=1)
    x = jnp.array(input_vec)
    
    # Run the model using the modified evaluation function.
    loss, x_hat = eval_on_batch_for_vis(T, x, model=model, optim_h=optim_h)
    
    # Create side-by-side plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Extract only the image portion (first 784 values) and reshape to (28, 28)
    orig_img = jnp.reshape(x[0, :784], (28, 28))
    ax1.imshow(jnp.clip(orig_img, 0.0, 1.0), cmap='gray')
    ax1.set_title(f'Original (Label: {label.item()})')
    ax1.axis('off')
    
    # x_hat is a 1D flattened vector (of length 794) where the first 784 values are the image.
    # Extract the image portion and reshape to (28, 28).
    img_rec = jnp.reshape(x_hat[:784], (28, 28))
    ax2.imshow(jnp.clip(img_rec, 0.0, 1.0), cmap='gray')
    ax2.set_title('Reconstruction')
    ax2.axis('off')
    
    plt.show()
    
    return orig_img, img_rec

import optax

if __name__ == '__main__':
    batch_size = 1
    nm_epochs = 10
    
    # For concatenated input: image (28x28 = 784) and label (10) -> 794.
    model = Decoder(
        input_dim=784 + 10, 
        hidden_dim=512, 
        output_dim=784 + 10, 
        nm_layers=2, 
        act_fn=jax.nn.swish
    )
    
    optim_h = pxu.Optim(lambda: optax.sgd(5e-2, momentum=0.1))
    optim_w = pxu.Optim(lambda: optax.adamw(1e-4), pxu.M(pxnn.LayerParam)(model))
    
    # Only use 100 samples for training and testing.
    train_dataloader, test_dataloader = get_dataloaders(batch_size, train_subset_n=100, test_subset_n=1000)
    
    for e in range(nm_epochs):
        train(train_dataloader, T=20, model=model, optim_w=optim_w, optim_h=optim_h)
        l = eval(test_dataloader, T=20, model=model, optim_h=optim_h)
        print(f"Epoch {e + 1}/{nm_epochs} - Test Loss: {l:.4f}")
    
    x_orig, x_recon = visualize_reconstruction(model, optim_h, T=20)