"""A differentiable copy of Diorama's renderer in PyTorch, for checking the WebGPU
backward pass (M2).

It renders a small random scene with the same math as the WGSL shaders
(preprocess.wgsl, tiles.wgsl, rasterize.wgsl), in float64. PyTorch's autograd gives
the exact gradients of a test loss. Those are checked here against finite
differences, then saved with the scene to src/checks/fixtures/backward.json. The
GPU check (src/checks/backward.ts) renders the same scene and compares its
gradients with them.

Run from the project root:
    .venv/Scripts/python reference/backward_reference.py     (Windows)
    .venv/bin/python reference/backward_reference.py         (macOS, Linux)
"""

import json
import math
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

WIDTH, HEIGHT = 32, 32  # two tiles by two, so splats cross tile edges
TILE = 16
NEAR_CULL = 0.2
BLUR = 0.3
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152789498]
SH_C3 = [
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
    -0.4570457994644658, 1.445305721320277, -0.5900435899266435,
]
SH_DEGREE = 3
FIXTURE = Path(__file__).resolve().parent.parent / "src" / "checks" / "fixtures" / "backward.json"


# 4×4 matrices as 16 numbers in column-major order, like src/mat4.ts.
def perspective(fov_y, aspect, near, far):
    f = 1 / math.tan(fov_y / 2)
    depth = 1 / (near - far)
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * depth, -1, 0, 0, near * far * depth, 0]


def look_at(eye, target, up):
    eye, target, up = (torch.tensor(v) for v in (eye, target, up))
    z = (eye - target) / (eye - target).norm()
    x = torch.linalg.cross(up, z)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    return [
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -x.dot(eye), -y.dot(eye), -z.dot(eye), 1,
    ]


def as_matrix(column_major):
    """Column-major numbers to a row-major 4×4 tensor."""
    return torch.tensor([float(v) for v in column_major]).reshape(4, 4).T


def rotation_matrices(raw):
    """(N, 4) quaternions (w, x, y, z), normalized first, to (N, 3, 3) matrices."""
    w, x, y, z = (raw / raw.norm(dim=1, keepdim=True)).unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)


def sh_basis(d):
    """(N, 3) unit directions to (N, 15) basis values, as in gaussianMath.wgsl."""
    x, y, z = d.unbind(1)
    xx, yy, zz = x * x, y * y, z * z
    return torch.stack([
        -SH_C1 * y, SH_C1 * z, -SH_C1 * x,
        SH_C2[0] * x * y, SH_C2[1] * y * z, SH_C2[2] * (2 * zz - xx - yy), SH_C2[3] * x * z, SH_C2[4] * (xx - yy),
        SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * x * y * z, SH_C3[2] * y * (4 * zz - xx - yy),
        SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy), SH_C3[4] * x * (4 * zz - xx - yy),
        SH_C3[5] * z * (xx - yy), SH_C3[6] * x * (xx - 3 * yy),
    ], dim=1)


def render(params, view, proj, background):
    """The renderer's forward pass: an (H·W, 3) image. Discrete choices (culling,
    tile coverage, thresholds, early stopping) are made without gradients, as on the GPU."""
    position, scale, rotation, opacity, color, sh = params
    w = view[:3, :3]
    t = position @ w.T + view[:3, 3]
    depth = -t[:, 2]

    r = rotation_matrices(rotation)
    m = r * scale[:, None, :]  # scales column k of R by scale k
    cov3d = m @ m.transpose(1, 2)

    fx = proj[0, 0] * WIDTH / 2
    fy = proj[1, 1] * HEIGHT / 2
    limit_x = 1.3 * WIDTH / 2 / fx
    limit_y = 1.3 * HEIGHT / 2 / fy
    tx = torch.clamp(t[:, 0] / depth, -limit_x, limit_x) * depth
    ty = torch.clamp(t[:, 1] / depth, -limit_y, limit_y) * depth
    zero = torch.zeros_like(depth)
    j = torch.stack([fx / depth, zero, fx * tx / depth**2, zero, -fy / depth, -fy * ty / depth**2], dim=1)
    jw = j.reshape(-1, 2, 3) @ w
    cov = jw @ cov3d @ jw.transpose(1, 2)
    a = cov[:, 0, 0] + BLUR
    b = cov[:, 1, 0]
    c = cov[:, 1, 1] + BLUR
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], dim=1)

    clip = torch.cat([t, torch.ones_like(depth)[:, None]], dim=1) @ proj.T
    ndc = clip[:, :2] / clip[:, 3:]
    size = torch.tensor([WIDTH, HEIGHT], dtype=torch.float64)
    mean = (ndc * torch.tensor([0.5, -0.5]) + 0.5) * size

    camera_position = -(w.T @ view[:3, 3])
    offset = position - camera_position
    direction = offset / offset.norm(dim=1, keepdim=True)
    count = (SH_DEGREE + 1) ** 2 - 1
    shaded = torch.clamp(color + (sh_basis(direction)[:, :count, None] * sh[:, :count, :]).sum(1), min=0)

    ys, xs = torch.meshgrid(torch.arange(HEIGHT), torch.arange(WIDTH), indexing="ij")
    pixel_x, pixel_y = xs.flatten(), ys.flatten()
    centers = torch.stack([pixel_x + 0.5, pixel_y + 0.5], dim=1).to(torch.float64)
    d = centers[:, None, :] - mean[None, :, :]  # pixel − center, (pixels, splats, 2)
    power = -0.5 * (conic[:, 0] * d[..., 0] ** 2 + conic[:, 2] * d[..., 1] ** 2) - conic[:, 1] * d[..., 0] * d[..., 1]
    alpha = torch.clamp(opacity * torch.exp(power), max=0.99)

    with torch.no_grad():
        mid = 0.5 * (a + c)
        radius = torch.ceil(3 * torch.sqrt(mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))))
        offscreen = ((mean + radius[:, None] < 0) | (mean - radius[:, None] > size)).any(dim=1)
        visible = (depth > NEAR_CULL) & (det > 0) & ~offscreen
        tiles = torch.tensor([math.ceil(WIDTH / TILE), math.ceil(HEIGHT / TILE)])
        low = torch.minimum(torch.clamp(torch.floor((mean - radius[:, None]) / TILE), min=0), tiles)
        high = torch.minimum(torch.clamp(torch.floor((mean + radius[:, None]) / TILE) + 1, min=0), tiles)
        tile_x, tile_y = (pixel_x // TILE)[:, None], (pixel_y // TILE)[:, None]
        covered = (tile_x >= low[:, 0]) & (tile_x < high[:, 0]) & (tile_y >= low[:, 1]) & (tile_y < high[:, 1])
        blended = covered & visible[None, :] & (power <= 0) & (alpha >= 1 / 255)

    # Front to back; a stable sort keeps equal depths in list order, as on the GPU.
    order = torch.argsort(depth.detach(), stable=True)
    alpha = torch.where(blended, alpha, 0)[:, order]
    with torch.no_grad():
        # A pixel stops at the splat that would take its transmittance below
        # 0.0001, without adding it.
        before = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], dim=1), dim=1)
        stops = blended[:, order] & (before * (1 - alpha) < 0.0001)
        used = torch.cumsum(stops.to(torch.int64), dim=1) == 0
    alpha = torch.where(used, alpha, 0)
    transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], dim=1), dim=1)
    final = transmittance[:, -1] * (1 - alpha[:, -1])
    return (alpha * transmittance) @ shaded[order] + final[:, None] * background


def main():
    generator = torch.Generator().manual_seed(2)

    def uniform(shape, low, high):
        return low + (high - low) * torch.rand(shape, generator=generator)

    n = 10
    params = [
        uniform((n, 3), -0.8, 0.8),  # position
        uniform((n, 3), 0.15, 0.5),  # scale
        torch.randn((n, 4), generator=generator),  # rotation, deliberately not unit length
        uniform((n,), 0.35, 0.85),  # opacity, kept under the 0.99 cap
        uniform((n, 3), 0.45, 0.75),  # base color, kept clear of the clamp at zero
        uniform((n, 15, 3), -0.05, 0.05),  # spherical harmonics
    ]
    view_list = [float(v) for v in look_at([1.2, 0.8, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])]
    proj_list = perspective(math.radians(50), WIDTH / HEIGHT, 0.1, 100)
    view, proj = as_matrix(view_list), as_matrix(proj_list)
    background = torch.tensor([0.1, 0.2, 0.3])
    # The test loss is Σ weight × pixel, so dL/d(pixel) is just the weights.
    weights = uniform((WIDTH * HEIGHT, 3), -1.0, 1.0)

    def loss(values):
        return (render(values, view, proj, background) * weights).sum()

    leaves = [p.clone().requires_grad_(True) for p in params]
    image = render(leaves, view, proj, background)
    (image * weights).sum().backward()
    gradients = [leaf.grad for leaf in leaves]

    # Check autograd against central finite differences, one number at a time.
    names = ["position", "scale", "rotation", "opacity", "color", "sh"]
    step = 1e-6
    finite_difference_error = {}
    for index, (name, value, gradient) in enumerate(zip(names, params, gradients)):
        flat = value.flatten()
        estimate = torch.zeros_like(flat)
        for k in range(flat.numel()):
            nudged = [p.clone() for p in params]
            nudged[index] = flat.clone()
            nudged[index][k] += step
            up = loss([p.reshape(q.shape) for p, q in zip(nudged, params)])
            nudged[index][k] -= 2 * step
            down = loss([p.reshape(q.shape) for p, q in zip(nudged, params)])
            estimate[k] = (up - down) / (2 * step)
        scale_of = gradient.abs().max().item()
        finite_difference_error[name] = (estimate - gradient.flatten()).abs().max().item() / scale_of
        print(f"{name:9s} largest gradient {scale_of:.3e}, finite differences disagree by {finite_difference_error[name]:.2e} of that")

    position, scale, rotation, opacity, color, sh = params
    fixture = {
        "about": "Made by reference/backward_reference.py. Gradients are PyTorch autograd, float64.",
        "viewport": [WIDTH, HEIGHT],
        "view": view_list,
        "proj": proj_list,
        "background": background.tolist(),
        "shDegree": SH_DEGREE,
        "gaussians": [
            {
                "position": position[i].tolist(),
                "scale": scale[i].tolist(),
                "rotation": rotation[i].tolist(),
                "color": color[i].tolist(),
                "opacity": opacity[i].item(),
            }
            for i in range(n)
        ],
        "shRest": [sh[i].flatten().tolist() for i in range(n)],  # coefficient-major, like SH_REST_FLOATS
        "pixelGrads": weights.flatten().tolist(),
        "image": image.detach().flatten().tolist(),
        "gradients": [
            {
                "position": gradients[0][i].tolist(),
                "scale": gradients[1][i].tolist(),
                "rotation": gradients[2][i].tolist(),
                "opacity": gradients[3][i].item(),
                "color": gradients[4][i].tolist(),
                "sh": gradients[5][i].flatten().tolist(),
            }
            for i in range(n)
        ],
        "finiteDifferenceError": finite_difference_error,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture))
    print(f"Wrote {FIXTURE}")


if __name__ == "__main__":
    main()
