import numpy as np

try:
    import torch

    has_torch = True
except ModuleNotFoundError:
    has_torch = False

if has_torch:
    torch_dtype_to_numpy = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
    }

    class TorchifiedSearchOriginGenerator:
        def __init__(self, generator, device):
            self.generator = generator
            self.device = device

        def __call__(self):
            return torch.as_tensor(self.generator()).to(self.device)

    class TorchifiedProjector:
        def __init__(self, projector):
            self.projector = projector

        def __call__(self, gradient):
            perm_mat = self.projector(gradient.cpu().numpy())
            return torch.as_tensor(perm_mat).to(gradient.device)
