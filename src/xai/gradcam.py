from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ForwardFn = Callable[..., torch.Tensor]


@dataclass
class GradCAMResult:
    cam: np.ndarray
    logit: float
    prob: float


class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        forward_fn: Optional[ForwardFn] = None,
        *,
        apply_sigmoid: bool = True,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.forward_fn = forward_fn
        self.apply_sigmoid = apply_sigmoid

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks_removed = False

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activations
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(
        self, module: nn.Module, inputs: Tuple[Any, ...], output: torch.Tensor
    ) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[Optional[torch.Tensor], ...],
    ) -> None:
        if grad_output[0] is None:
            raise RuntimeError("Grad-CAM backward hook received None gradients.")
        self.gradients = grad_output[0].detach()

    def _forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.forward_fn is not None:
            return self.forward_fn(self.model, *inputs)
        return self.model(*inputs)

    @staticmethod
    def _spatial_size(inputs: Sequence[torch.Tensor]) -> Tuple[int, int]:
        for tensor in inputs:
            if tensor.ndim >= 2:
                return int(tensor.shape[-2]), int(tensor.shape[-1])
        raise ValueError("Could not infer spatial size from Grad-CAM inputs.")

    @staticmethod
    def _scalar_score(logit: torch.Tensor, target_index: int = 0) -> torch.Tensor:
        if logit.ndim == 0:
            return logit
        flat = logit.reshape(-1)
        if flat.numel() == 0:
            raise ValueError("Model returned an empty logit tensor.")
        idx = min(target_index, flat.numel() - 1)
        return flat[idx]

    def generate(
        self,
        *inputs: torch.Tensor,
        target_index: int = 0,
        resize_to: Optional[Tuple[int, int]] = None,
    ) -> GradCAMResult:
        if self._hooks_removed:
            raise RuntimeError("GradCAM hooks were already removed; create a new instance.")
        if not inputs:
            raise ValueError("GradCAM.generate requires at least one input tensor.")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        logit = self._forward(*inputs)
        score = self._scalar_score(logit, target_index=target_index)
        score.backward()

        activations = self.activations
        gradients = self.gradients
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        size = resize_to if resize_to is not None else self._spatial_size(inputs)
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)
        cam_min = float(cam_np.min())
        cam_max = float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        logit_value = float(score.detach().cpu().item())
        if self.apply_sigmoid:
            prob = float(torch.sigmoid(score.detach()).cpu().item())
        else:
            prob = logit_value

        return GradCAMResult(cam=cam_np, logit=logit_value, prob=prob)

    def close(self) -> None:
        if self._hooks_removed:
            return
        self.forward_handle.remove()
        self.backward_handle.remove()
        self._hooks_removed = True

    def remove_hooks(self) -> None:
        self.close()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
