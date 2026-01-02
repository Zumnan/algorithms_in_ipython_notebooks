import torch
import torch.nn.functional as F

# Default/static weights
ALPHA_DICE = 0.3
ALPHA_ORTHO = 0.05
ALPHA_ALIGN = 0.05

# Warmup controls (ramp dice/align over first K steps)
WARMUP_STEPS = 2000

# Optional focal-ize CE
USE_FOCAL_CE = False
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_FG = 0.75


class TextDecomposer(torch.nn.Module):
    """
    Two learnable heads that map a sentence embedding to two subspaces for Lortho.
    """

    def __init__(self, dim_in: int, dim_out: int = 256):
        super().__init__()
        self.proj1 = torch.nn.Sequential(
            torch.nn.LayerNorm(dim_in),
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_out, dim_out),
        )
        self.proj2 = torch.nn.Sequential(
            torch.nn.LayerNorm(dim_in),
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_out, dim_out),
        )

    def forward(self, sent_vec: torch.Tensor):
        t1 = self.proj1(sent_vec)
        t2 = self.proj2(sent_vec)
        return t1, t2


class AlignLite(torch.nn.Module):
    """
    Aligns a global mask descriptor and sentence embedding with cosine loss.
    """

    def __init__(self, txt_dim: int, hid: int = 256):
        super().__init__()
        self.txt_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(txt_dim),
            torch.nn.Linear(txt_dim, hid),
            torch.nn.ReLU(True),
            torch.nn.Linear(hid, hid),
        )
        self.mask_proj = torch.nn.Sequential(
            torch.nn.Linear(4, hid),
            torch.nn.ReLU(True),
            torch.nn.Linear(hid, hid),
        )

    @staticmethod
    def mask_descriptor(mask_fg: torch.Tensor):
        B, H, W = mask_fg.shape
        area = mask_fg.sum(dim=(1, 2)).clamp_min(1e-6)
        ratio = area / float(H * W)

        ys = torch.linspace(0, 1, steps=H, device=mask_fg.device).view(1, H, 1)
        xs = torch.linspace(0, 1, steps=W, device=mask_fg.device).view(1, 1, W)
        cy = (mask_fg * ys).sum(dim=(1, 2)) / area
        cx = (mask_fg * xs).sum(dim=(1, 2)) / area

        top = mask_fg[:, 0, :].sum(dim=1)
        bot = mask_fg[:, -1, :].sum(dim=1)
        lef = mask_fg[:, :, 0].sum(dim=1)
        rig = mask_fg[:, :, -1].sum(dim=1)
        boundary = (top + bot + lef + rig) / (area + 1e-6)
        return torch.stack([ratio, cy, cx, boundary], dim=1)

    def forward(self, last_hidden: torch.Tensor, attn_mask: torch.Tensor, logits: torch.Tensor):
        s = sentence_mean_pool(last_hidden, attn_mask)
        s = self.txt_proj(s)

        pf = torch.softmax(logits, dim=1)[:, 1]
        desc = self.mask_descriptor(pf)
        v = self.mask_proj(desc)

        s = torch.nn.functional.normalize(s, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)
        cos = (s * v).sum(dim=-1)
        return (1.0 - cos).mean()


def focal_ce_with_logits(logits, target, gamma=2.0, alpha_fg=0.75):
    if logits.shape[1] != 2:
        raise ValueError("focal_ce_with_logits assumes 2-class logits.")
    log_prob = F.log_softmax(logits, dim=1)
    prob = torch.exp(log_prob)
    tgt = target.long()
    log_pt = log_prob.gather(1, tgt.unsqueeze(1)).squeeze(1)
    pt = prob.gather(1, tgt.unsqueeze(1)).squeeze(1)

    alpha = torch.ones_like(pt) * (1.0 - alpha_fg)
    alpha = torch.where(tgt == 1, torch.ones_like(pt) * alpha_fg, alpha)

    loss = -alpha * ((1 - pt) ** gamma) * log_pt
    return loss.mean()


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]
    target_f = (target > 0).float()
    if probs.shape[-2:] != target_f.shape[-2:]:
        probs = torch.nn.functional.interpolate(
            probs.unsqueeze(1), size=target_f.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
    intersection = (probs * target_f).sum(dim=(-1, -2))
    denom = probs.sum(dim=(-1, -2)) + target_f.sum(dim=(-1, -2))
    dice = (2 * intersection + eps) / (denom + eps)
    return (1.0 - dice).mean()


def squared_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    a = torch.nn.functional.normalize(a, dim=-1, eps=eps)
    b = torch.nn.functional.normalize(b, dim=-1, eps=eps)
    cos = (a * b).sum(dim=-1)
    return (cos ** 2).mean()


def sentence_mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor):
    mask = attn_mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (last_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
    return pooled


def composite_loss(
    output_logits,
    target,
    last_hidden_states=None,
    attn_mask=None,
    text_decomposer=None,
    align_lite=None,
    alpha_dice: float = ALPHA_DICE,
    alpha_ortho: float = ALPHA_ORTHO,
    alpha_align: float = ALPHA_ALIGN,
):
    if USE_FOCAL_CE:
        l_ce = focal_ce_with_logits(output_logits, target, gamma=FOCAL_GAMMA, alpha_fg=FOCAL_ALPHA_FG)
    else:
        ce_w = torch.tensor([0.9, 1.1], device=output_logits.device)
        l_ce = F.cross_entropy(output_logits, target, weight=ce_w)
    l_dice = dice_loss_from_logits(output_logits, target)
    l_seg = l_ce + alpha_dice * l_dice

    l_ortho = torch.tensor(0.0, device=output_logits.device)
    l_align = torch.tensor(0.0, device=output_logits.device)

    if (last_hidden_states is not None) and (attn_mask is not None) and (text_decomposer is not None):
        s = sentence_mean_pool(last_hidden_states, attn_mask)
        t1, t2 = text_decomposer(s)
        l_ortho = squared_cosine_similarity(t1, t2)

    if (last_hidden_states is not None) and (attn_mask is not None) and (align_lite is not None):
        l_align = align_lite(last_hidden_states, attn_mask, output_logits)

    total = l_seg + alpha_ortho * l_ortho + alpha_align * l_align
    return total, {
        "l_ce": l_ce.detach(),
        "l_dice": l_dice.detach(),
        "l_ortho": l_ortho.detach(),
        "l_align": l_align.detach(),
        "l_seg": l_seg.detach(),
    }
