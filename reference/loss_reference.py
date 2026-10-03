"""The training loss in PyTorch, for checking the WebGPU loss and its gradient (M3).

The loss is 0.8 × L1 + 0.2 × (1 − SSIM), with SSIM computed the way the original
3DGS code does (utils/loss_utils.py: an 11×11 Gaussian window with σ = 1.5, zero
padding, and the mean over every pixel and channel). This script takes a random
render and target in float64, gets the loss's gradient from autograd, checks it
against finite differences, and saves everything to src/checks/fixtures/loss.json.

Run from the project root:
    .venv/Scripts/python reference/loss_reference.py     (Windows)
    .venv/bin/python reference/loss_reference.py         (macOS, Linux)
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

WIDTH, HEIGHT = 37, 23  # odd sizes, so nothing lines up with the 16×16 dispatch
LAMBDA = 0.2
FIXTURE = Path(__file__).resolve().parent.parent / "src" / "checks" / "fixtures" / "loss.json"


def window(size=11, sigma=1.5):
    g = torch.tensor([math.exp(-((x - size // 2) ** 2) / (2 * sigma**2)) for x in range(size)])
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a, b):
    """The reference's SSIM: a and b are (3, H, W)."""
    w = window()
    a, b = a[None], b[None]
    blur = lambda image: F.conv2d(image, w, padding=5, groups=3)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return s.mean()


def loss(render, target):
    return (1 - LAMBDA) * (render - target).abs().mean() + LAMBDA * (1 - ssim(render, target))


def main():
    generator = torch.Generator().manual_seed(4)
    # The target is whole steps of 1/255, so an 8-bit texture holds it exactly.
    target = torch.randint(0, 256, (3, HEIGHT, WIDTH), generator=generator) / 255
    # The render is the target plus noise, like a half-trained model, so SSIM is
    # somewhere in the middle and both terms matter.
    render = (target + 0.15 * torch.randn((3, HEIGHT, WIDTH), generator=generator)).clamp(0, 1)
    render.requires_grad_(True)
    value = loss(render, target)
    value.backward()
    gradient = render.grad

    step = 1e-6
    estimate = torch.zeros_like(gradient)
    flat = render.detach().flatten()
    for k in range(flat.numel()):
        up, down = flat.clone(), flat.clone()
        up[k] += step
        down[k] -= step
        estimate.view(-1)[k] = (loss(up.view_as(render), target) - loss(down.view_as(render), target)) / (2 * step)
    disagreement = ((estimate - gradient).abs().max() / gradient.abs().max()).item()
    print(f"loss {value.item():.6f}, SSIM {ssim(render.detach(), target).item():.4f}")
    print(f"autograd and finite differences disagree by {disagreement:.2e} of the largest gradient")

    # Pixels row by row, rgb each, like the GPU's buffers.
    as_pixels = lambda image: image.permute(1, 2, 0).flatten().tolist()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({
        "about": "Made by reference/loss_reference.py. The gradient is PyTorch autograd, float64.",
        "size": [WIDTH, HEIGHT],
        "render": as_pixels(render.detach()),
        "target": [round(v * 255) for v in as_pixels(target)],
        "loss": value.item(),
        "gradient": as_pixels(gradient),
        "finiteDifferenceError": disagreement,
    }))
    print(f"Wrote {FIXTURE}")


if __name__ == "__main__":
    main()
