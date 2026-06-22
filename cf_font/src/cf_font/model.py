from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContentEncoder(nn.Module):
    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            ConvBlock(1, c),
            ConvBlock(c, c * 2, stride=2),
            ConvBlock(c * 2, c * 4, stride=2),
            ConvBlock(c * 4, c * 4, stride=2),
        )
        self.out_channels = c * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.forward(x)
        return F.adaptive_avg_pool2d(feature, (1, 1)).flatten(1)


class HintEncoder(nn.Module):
    def __init__(self, in_channels: int = 7, out_channels: int = 96) -> None:
        super().__init__()
        c = max(16, out_channels // 4)
        self.net = nn.Sequential(
            ConvBlock(in_channels, c),
            ConvBlock(c, c * 2, stride=2),
            ConvBlock(c * 2, out_channels, stride=2),
            ConvBlock(out_channels, out_channels, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StyleEncoder(nn.Module):
    def __init__(self, style_dim: int = 96, base_channels: int = 24) -> None:
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            ConvBlock(1, c),
            ConvBlock(c, c * 2, stride=2),
            ConvBlock(c * 2, c * 4, stride=2),
            ConvBlock(c * 4, c * 4, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(c * 4, style_dim)

    def forward(self, refs: torch.Tensor) -> torch.Tensor:
        if refs.dim() != 5:
            raise ValueError(f"style refs must be B,Q,1,H,W, got {refs.shape}")
        batch, ref_count = refs.shape[:2]
        flat = refs.reshape(batch * ref_count, *refs.shape[2:])
        vectors = self.head(self.features(flat).flatten(1))
        return vectors.reshape(batch, ref_count, -1).mean(dim=1)


class ContentFusionModule(nn.Module):
    """CF-Font content fusion: weighted sum of basis content features."""

    def __init__(self, temperature: float = 0.18) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def weights_from_distances(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.softmax(-distances / max(self.temperature, 1e-6), dim=1)

    def forward(self, basis_features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if basis_features.dim() != 5:
            raise ValueError(f"basis features must be B,K,C,H,W, got {basis_features.shape}")
        if weights.shape[:2] != basis_features.shape[:2]:
            raise ValueError(f"weights {weights.shape} do not match basis {basis_features.shape}")
        return (basis_features * weights[:, :, None, None, None]).sum(dim=1)


class Decoder(nn.Module):
    def __init__(self, channels: int, base_channels: int = 24) -> None:
        super().__init__()
        c = base_channels
        self.mix = ConvBlock(channels * 2, channels)
        self.up3 = nn.ConvTranspose2d(channels, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 4, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 2, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c, c)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, content: torch.Tensor, hints: torch.Tensor) -> torch.Tensor:
        x = self.mix(torch.cat([content, hints], dim=1))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return self.head(x)


class CFFontEcoNet(nn.Module):
    def __init__(
        self,
        base_channels: int = 24,
        style_dim: int = 96,
        num_eco_styles: int = 4,
        hint_channels: int = 7,
        cfm_temperature: float = 0.18,
    ) -> None:
        super().__init__()
        self.content_encoder = ContentEncoder(base_channels=base_channels)
        self.hint_encoder = HintEncoder(in_channels=hint_channels, out_channels=self.content_encoder.out_channels)
        self.style_encoder = StyleEncoder(style_dim=style_dim, base_channels=base_channels)
        self.eco_style_embedding = nn.Embedding(num_eco_styles, style_dim)
        self.saving_mlp = nn.Sequential(nn.Linear(1, style_dim), nn.SiLU(inplace=True), nn.Linear(style_dim, style_dim))
        self.condition_mlp = nn.Sequential(nn.Linear(style_dim * 3, style_dim), nn.SiLU(inplace=True), nn.Linear(style_dim, style_dim))
        self.affine = nn.Linear(style_dim, self.content_encoder.out_channels * 2)
        self.decoder = Decoder(channels=self.content_encoder.out_channels, base_channels=base_channels)
        self.cfm = ContentFusionModule(temperature=cfm_temperature)
        self.model_config = {
            "base_channels": base_channels,
            "style_dim": style_dim,
            "num_eco_styles": num_eco_styles,
            "hint_channels": hint_channels,
            "cfm_temperature": cfm_temperature,
        }

    def encode_style_refs(self, style_refs: torch.Tensor) -> torch.Tensor:
        return self.style_encoder(style_refs)

    def condition_vector(self, font_style: torch.Tensor, eco_style_id: torch.Tensor, target_saving: torch.Tensor) -> torch.Tensor:
        saving = target_saving.reshape(-1, 1).to(font_style.dtype)
        eco = self.eco_style_embedding(eco_style_id)
        saving_vec = self.saving_mlp(saving)
        return self.condition_mlp(torch.cat([font_style, eco, saving_vec], dim=1))

    def modulate(self, content: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        mean = content.mean(dim=(2, 3), keepdim=True)
        std = content.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        return (1.0 + gamma) * (content - mean) / std + beta

    def decode_from_features(
        self,
        content_feature: torch.Tensor,
        hints: torch.Tensor,
        font_style: torch.Tensor,
        eco_style_id: torch.Tensor,
        target_saving: torch.Tensor,
    ) -> torch.Tensor:
        hint_feature = self.hint_encoder(hints)
        condition = self.condition_vector(font_style, eco_style_id, target_saving)
        content = self.modulate(content_feature, condition)
        return self.decoder(content, hint_feature)

    def forward_base(
        self,
        content: torch.Tensor,
        style_refs: torch.Tensor,
        hints: torch.Tensor,
        eco_style_id: torch.Tensor,
        target_saving: torch.Tensor,
    ) -> torch.Tensor:
        content_feature = self.content_encoder(content)
        font_style = self.encode_style_refs(style_refs)
        return self.decode_from_features(content_feature, hints, font_style, eco_style_id, target_saving)

    def content_feature_from_basis(self, basis_images: torch.Tensor, cfm_weights: torch.Tensor) -> torch.Tensor:
        if basis_images.dim() != 5:
            raise ValueError(f"basis images must be B,K,1,H,W, got {basis_images.shape}")
        batch, basis_count = basis_images.shape[:2]
        flat = basis_images.reshape(batch * basis_count, *basis_images.shape[2:])
        encoded = self.content_encoder(flat)
        features = encoded.reshape(batch, basis_count, self.content_encoder.out_channels, *encoded.shape[-2:])
        return self.cfm(features, cfm_weights)

    def forward_cf(
        self,
        basis_images: torch.Tensor,
        cfm_weights: torch.Tensor,
        style_refs: torch.Tensor,
        hints: torch.Tensor,
        eco_style_id: torch.Tensor,
        target_saving: torch.Tensor,
    ) -> torch.Tensor:
        content_feature = self.content_feature_from_basis(basis_images, cfm_weights)
        font_style = self.encode_style_refs(style_refs)
        return self.decode_from_features(content_feature, hints, font_style, eco_style_id, target_saving)

    @torch.no_grad()
    def content_embeddings(self, images: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        chunks = []
        for start in range(0, images.size(0), batch_size):
            chunks.append(self.content_encoder.embedding(images[start : start + batch_size]))
        return torch.cat(chunks, dim=0)
