"""
MOT SUTrack - Motion-aware Object Tracking Temporal Modules

Three integrated modules for temporal and motion modeling:
1. TemplateTemporalEncoder (TTE): Models temporal dynamics across multi-template frames
2. SearchTemplateCrossAttention (STCA): Motion-guided cross-attention from search to temporal templates
3. TrajectoryMemoryBank (TMB): Long-term memory of target appearance and motion priors

For Anti-UAV infrared drone tracking: drones are small, fast-moving,
and often lost in IR noise. Temporal/motion information is critical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemplateTemporalEncoder(nn.Module):
    """
    Models temporal dynamics across multiple template frames.

    Input: template features from different time points (B, N_t, L_t, C)
    Output: temporally enriched template features (B, L_t, C)

    Key insight: The template frames are sampled from different timestamps.
    Their feature differences encode appearance changes and implicit motion.
    By modeling their temporal relationship, we extract motion-aware features
    that help localize the target in the current search frame.

    Architecture:
    1. Temporal Position Encoding: Distinguish frames by their relative time offset
    2. Temporal Self-Attention: Across frames, per spatial position
    3. Temporal Convolution: Capture local temporal patterns
    4. Frame-wise fusion: Weighted aggregation into a single template representation
    """

    def __init__(self, embed_dim=512, num_heads=8, num_frames=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, 1, embed_dim)
        )

        self.temporal_norm1 = nn.LayerNorm(embed_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.temporal_norm2 = nn.LayerNorm(embed_dim)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
        )

        self.frame_weight_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, 1),
        )

        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

    def forward(self, template_feats):
        """
        Args:
            template_feats: (B, N_t, L_t, C) features from N_t template frames
        Returns:
            temporal_template: (B, L_t, C) temporally aggregated template features
            attention_weights: temporal attention weights for analysis
        """
        B, N_t, L_t, C = template_feats.shape

        if N_t == 1:
            return template_feats.squeeze(1), None

        feat = template_feats + self.temporal_pos_embed[:, :N_t, :, :]

        feat_reshaped = feat.permute(0, 2, 1, 3).reshape(B * L_t, N_t, C)

        attn_out, attn_weights = self.temporal_attn(
            feat_reshaped, feat_reshaped, feat_reshaped
        )
        feat_reshaped = self.temporal_norm1(feat_reshaped + self.dropout(attn_out))
        feat_reshaped = feat_reshaped.reshape(B, L_t, N_t, C)

        feat_transposed = feat_reshaped.reshape(B * L_t, N_t, C).permute(0, 2, 1)
        conv_out = self.temporal_conv(feat_transposed)
        conv_out = conv_out.permute(0, 2, 1).reshape(B, N_t, L_t, C).permute(0, 2, 1, 3)
        feat_reshaped = self.temporal_norm2(feat_reshaped + self.dropout(conv_out))

        frame_pooled = feat_reshaped.mean(dim=1)
        frame_weights = self.frame_weight_net(frame_pooled).squeeze(-1)
        frame_weights = frame_weights.softmax(dim=-1)

        temporal_template = (feat_reshaped * frame_weights.unsqueeze(1).unsqueeze(-1)).sum(dim=2)

        return temporal_template, (attn_weights, frame_weights)


class SearchTemplateCrossAttention(nn.Module):
    """
    Motion-guided Cross-Attention from search region to temporal templates (with internal gated residual).
    Used by MOT v1/v2. Output includes original search features via internal residual.
    """

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.gate = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, search_feat, template_feat, template_mask=None):
        attn_out, attn_weights = self.cross_attn(
            query=search_feat,
            key=template_feat,
            value=template_feat,
            key_padding_mask=template_mask,
        )

        gate_val = self.gate.sigmoid()
        search_feat = self.norm1(
            search_feat + self.dropout(attn_out) * gate_val
        )

        search_feat = self.norm2(
            search_feat + self.ffn(search_feat)
        )

        return search_feat, attn_weights


class SearchTemplateCrossAttentionDelta(nn.Module):
    """
    Pure delta version of STCA for RTI (Residual Temporal Injection).

    Outputs ONLY the temporal delta (no internal self-residual).
    The caller applies: final = search + gate * delta

    This ensures gate=0 → exact Standard baseline, no double-counting.
    Added parameters vs Standard: ~0 (same as STCA), inference FLOPs: identical.
    """

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, search_feat, template_feat, template_mask=None):
        attn_out, attn_weights = self.cross_attn(
            query=search_feat,
            key=template_feat,
            value=template_feat,
            key_padding_mask=template_mask,
        )
        delta = self.norm(self.dropout(attn_out) + self.ffn(attn_out))
        return delta, attn_weights


class TrajectoryMemoryBank(nn.Module):
    """
    Maintains a learnable memory bank of target features and position history.

    Purpose:
    1. Provides long-term appearance memory for re-detection after occlusion
    2. Computes motion priors (velocity, acceleration) from position history
    3. Feature consistency regularization across frames

    The memory bank stores K most recent target states, each containing:
    - feature vector (from the target region in the search frame)
    - bbox prediction
    - timestamp/age

    A time-decayed weighted readout provides context for the current frame.
    """

    def __init__(self, feature_dim=512, memory_size=8, decay_rate=0.92):
        super().__init__()
        self.memory_size = memory_size
        self.decay_rate = decay_rate

        self.register_buffer('features', torch.zeros(memory_size, feature_dim))
        self.register_buffer('bboxes', torch.zeros(memory_size, 4))
        self.register_buffer('ages', torch.zeros(memory_size))
        self.register_buffer('ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('filled', torch.tensor(False))

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        self.motion_predictor = nn.Sequential(
            nn.Linear(feature_dim + 4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 2),
        )

    def update(self, feat, bbox):
        """
        Update memory bank with new target observation.
        feat: (B, C) target feature, averaged over batch
        bbox: (B, 4) predicted bbox, averaged over batch
        """
        idx = self.ptr % self.memory_size
        self.features[idx] = feat.detach().mean(0)
        self.bboxes[idx] = bbox.detach().mean(0)
        self.ages[idx] = 0.0
        self.ptr += 1
        if self.ptr >= self.memory_size:
            self.filled = torch.tensor(True)

    def age_and_read(self, current_feat):
        """
        Age all entries and read context.
        current_feat: (B, C) current search target region feature
        Returns: memory_context (B, C), velocity_prior (B, 2)
        """
        self.ages += 1
        if not self.filled:
            return (torch.zeros_like(current_feat),
                    torch.zeros(current_feat.size(0), 2, device=current_feat.device))

        time_weights = (self.decay_rate ** self.ages).to(current_feat.device)
        time_weights = time_weights / (time_weights.sum() + 1e-8)

        memory_context = (self.features * time_weights.unsqueeze(1)).sum(0)
        memory_context = self.feature_proj(memory_context)
        memory_context = memory_context.unsqueeze(0).expand(current_feat.size(0), -1)

        sorted_idx = torch.argsort(self.ages)
        recent_bboxes = self.bboxes[sorted_idx[:2]]
        c1 = recent_bboxes[0, :2] + recent_bboxes[0, 2:4] / 2
        c2 = recent_bboxes[1, :2] + recent_bboxes[1, 2:4] / 2
        dt = (self.ages[sorted_idx[1]] - self.ages[sorted_idx[0]]).clamp(min=1)
        velocity = (c2 - c1) / dt
        velocity = velocity.unsqueeze(0).expand(current_feat.size(0), -1)

        motion_input = torch.cat([memory_context, velocity], dim=-1)
        velocity_refined = self.motion_predictor(motion_input)

        return memory_context, velocity_refined


class SpatialFeatureExtractor(nn.Module):
    """
    Extracts target-centric feature from the search region features
    using the decoded spatial map. This is used to update the memory bank.
    """

    def __init__(self, embed_dim=512):
        super().__init__()
        self.attn_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, search_feat, score_map=None):
        """
        search_feat: (B, L, C) search region tokens
        score_map: (B, 1, H, W) predicted score map (optional, for guidance)
        Returns: target_feat (B, C)
        """
        if score_map is not None:
            B, _, H, W = score_map.shape
            score_flat = score_map.flatten(1).unsqueeze(-1)
            weights = score_flat / (score_flat.sum(dim=1, keepdim=True) + 1e-8)
            target_feat = (search_feat * weights).sum(dim=1)
        else:
            attn_weights = self.attn_pool(search_feat).softmax(dim=1)
            target_feat = (search_feat * attn_weights).sum(dim=1)

        return target_feat
