import jax
from jax.lib import xla_bridge

# Check backend platform
print("JAX backend:", xla_bridge.get_backend().platform)

# List available devices with full details
print("\nDetailed devices:")
for i, device in enumerate(jax.devices()):
    print(f"Device {i}:")
    print(f"  Platform: {device.platform}")
    print(f"  Device Type: {device.device_kind}")
    print(f"  Client ID: {device.client}")
    print(f"  Process Index: {device.process_index}")

import jax.numpy as jnp

# Explicit GPU placement test
x = jnp.array([1.0], device=jax.devices('gpu')[0])
print(x + x)  # Should show device=<jax.devices('gpu')[0]>

