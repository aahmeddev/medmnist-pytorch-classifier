"""Pure-PyTorch Grad-CAM implementation for convolutional explainability."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Compute class-discriminative Grad-CAM heatmaps using PyTorch hooks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module: nn.Module, _inp: tuple, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module: nn.Module, _grad_in: tuple, grad_out: tuple) -> None:
        self._gradients = grad_out[0].detach()

    def remove_hooks(self) -> None:
        """Detach registered PyTorch hooks cleanly."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, image: torch.Tensor, class_idx: Optional[int] = None) -> tuple[torch.Tensor, int, float]:
        """Generate a normalised [0, 1] Grad-CAM heatmap upsampled to the input image size."""
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("GradCAM.generate expects a single-image batch of shape (1, C, H, W).")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(image)
        probs = torch.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, class_idx].item())

        score = logits[0, class_idx]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)

        cam_min = cam.min()
        cam_max = cam.max()
        if float(cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.squeeze(0).squeeze(0).cpu(), class_idx, confidence
