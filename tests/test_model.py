"""Unit tests for MedResNet architecture and Grad-CAM explainability."""

import torch

from src.gradcam import GradCAM
from src.model import MedResNet


def test_medresnet_forward_shapes() -> None:
    model_gray = MedResNet(in_channels=1, num_classes=2)
    out_gray = model_gray(torch.randn(4, 1, 28, 28))
    assert out_gray.shape == (4, 2)

    model_rgb = MedResNet(in_channels=3, num_classes=9)
    out_rgb = model_rgb(torch.randn(4, 3, 28, 28))
    assert out_rgb.shape == (4, 9)


def test_gradcam_heatmap_bounds_and_shape() -> None:
    model = MedResNet(in_channels=1, num_classes=2)
    cam = GradCAM(model, model.target_conv_layer)
    heatmap, pred_idx, conf = cam.generate(torch.randn(1, 1, 28, 28))
    cam.remove_hooks()

    assert heatmap.shape == (28, 28)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0 + 1e-6
    assert pred_idx in (0, 1)
    assert 0.0 <= conf <= 1.0


def test_training_step_updates_weights() -> None:
    model = MedResNet(in_channels=1, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    before = model.stage3.conv2.weight.detach().clone()
    logits = model(torch.randn(8, 1, 28, 28))
    loss = criterion(logits, torch.randint(0, 2, (8,)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = model.stage3.conv2.weight.detach()
    assert not torch.allclose(before, after)
