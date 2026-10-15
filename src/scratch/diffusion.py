import torch

class GaussianDiffusion:
    def __init__(self, steps=1000, device="cpu"):
        self.steps = steps
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alpha_bar[t].sqrt()[:, None, None, None]
        b = (1 - self.alpha_bar[t]).sqrt()[:, None, None, None]
        return a * x0 + b * noise, noise

    @torch.no_grad()
    def sample(self, model, input_ids, shape, device):
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            pred_noise = model(x, t, input_ids)
            beta = self.betas[i]
            alpha = self.alphas[i]
            abar = self.alpha_bar[i]

            mean = (1 / alpha.sqrt()) * (x - beta / (1 - abar).sqrt() * pred_noise)
            if i > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)
