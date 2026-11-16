import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReverse.apply(x, lambd)


class Encoder_overall(Module):
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0,
                 act=F.relu, encoder_type='gcn', decoder_hidden_mult_omics1=2.0,
                 decoder_hidden_mult_omics2=1.5, use_residual_fusion=True, final_fusion_mode=None,
                 common_fusion_mode='conv_gate', private_fusion_mode='attention'):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act
        self.encoder_type = encoder_type
        self.decoder_hidden_mult_omics1 = decoder_hidden_mult_omics1
        self.decoder_hidden_mult_omics2 = decoder_hidden_mult_omics2
        self.use_residual_fusion = use_residual_fusion
        self.final_fusion_mode = final_fusion_mode or ('residual' if use_residual_fusion else 'shared')
        self.common_fusion_mode = common_fusion_mode
        if private_fusion_mode not in ('attention', 'concat'):
            raise ValueError(
                f"Unsupported private_fusion_mode: {private_fusion_mode}. "
                "Expected attention or concat."
            )
        self.private_fusion_mode = private_fusion_mode
        if self.encoder_type in ('gcn', 'vanilla_gcn'):
            encoder_cls = Encoder
        elif self.encoder_type in ('res_gcn', 'residual_gcn'):
            encoder_cls = ResidualGCNEncoder
        elif self.encoder_type == 'sage':
            encoder_cls = GraphSAGEEncoder
        elif self.encoder_type == 'gat':
            encoder_cls = GraphAttentionEncoder
        elif self.encoder_type in ('graph_transformer', 'transformer'):
            encoder_cls = GraphTransformerEncoder
        elif self.encoder_type in ('graph_mamba', 'mamba'):
            encoder_cls = GraphMambaEncoder
        else:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")

        self.encoder_omics1 = encoder_cls(
            self.dim_in_feat_omics1, self.dim_out_feat_omics1, dropout=self.dropout, act=self.act
        )
        self.decoder_omics1 = Decoder(
            self.dim_out_feat_omics1,
            self.dim_in_feat_omics1,
            hidden_mult=self.decoder_hidden_mult_omics1,
            dropout=self.dropout
        )
        self.encoder_omics2 = encoder_cls(
            self.dim_in_feat_omics2, self.dim_out_feat_omics2, dropout=self.dropout, act=self.act
        )
        self.decoder_omics2 = Decoder(
            self.dim_out_feat_omics2,
            self.dim_in_feat_omics2,
            hidden_mult=self.decoder_hidden_mult_omics2,
            dropout=self.dropout
        )

        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_cross = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)
        self.atten_shared = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)
        self.atten_private = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)
        self.shared_mamba_fusion = ModalityAwareMambaFusion(
            self.dim_out_feat_omics1,
            dropout=self.dropout,
            fusion_mode=self.common_fusion_mode,
        )

        self.shared_proj_omics1 = nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.shared_proj_omics2 = nn.Linear(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.private_proj_omics1 = nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.private_proj_omics2 = nn.Linear(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.dim_out_feat_omics1 * 2, self.dim_out_feat_omics1),
            nn.ReLU(),
            nn.Linear(self.dim_out_feat_omics1, 1)
        )
        self.residual_adapter = nn.Sequential(
            nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1),
            nn.ReLU(),
            nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        )
        self.concat_fusion_adapter = nn.Sequential(
            nn.Linear(self.dim_out_feat_omics1 * 2, self.dim_out_feat_omics1),
            nn.ReLU(),
            nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        )
        self.recon_fusion_omics1 = nn.Sequential(
            nn.Linear(self.dim_out_feat_omics1 + self.dim_out_feat_omics1, self.dim_out_feat_omics1),
            nn.ReLU(),
            nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        )
        self.recon_fusion_omics2 = nn.Sequential(
            nn.Linear(self.dim_out_feat_omics2 + self.dim_out_feat_omics2, self.dim_out_feat_omics2),
            nn.ReLU(),
            nn.Linear(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        )
        self.disc_shared = ModalityDiscriminator(self.dim_out_feat_omics1)
        self.disc_private = ModalityDiscriminator(self.dim_out_feat_omics1)

        # Ablation-only adapters are declared after every original module so
        # enabling the new modes does not alter Full-model initialization.
        rng_state = torch.get_rng_state()
        self.shared_concat_adapter = nn.Sequential(
            nn.Linear(self.dim_out_feat_omics1 * 2, self.dim_out_feat_omics1),
            nn.ReLU(),
            nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        )
        self.private_concat_adapter = nn.Sequential(
            nn.Linear(self.dim_out_feat_omics1 * 2, self.dim_out_feat_omics1),
            nn.ReLU(),
            nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        )
        torch.set_rng_state(rng_state)

    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2,
                adj_feature_omics2, adj_multi_level=None, adj_context_omics1=None, adj_context_omics2=None,
                spatial_order=None):
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)

        if adj_multi_level is None:
            adj_context_omics1 = adj_feature_omics1 if adj_context_omics1 is None else adj_context_omics1
            adj_context_omics2 = adj_feature_omics2 if adj_context_omics2 is None else adj_context_omics2
        else:
            adj_context_omics1 = adj_multi_level
            adj_context_omics2 = adj_multi_level

        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_context_omics1)
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_context_omics2)

        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)
        emb_anchor, alpha_anchor_modal = self.atten_cross(emb_latent_omics1, emb_latent_omics2)

        emb_shared_omics1 = self.shared_proj_omics1(emb_latent_omics1)
        emb_shared_omics2 = self.shared_proj_omics2(emb_latent_omics2)
        emb_private_omics1 = self.private_proj_omics1(emb_latent_omics1)
        emb_private_omics2 = self.private_proj_omics2(emb_latent_omics2)
        emb_shared_for_disc = torch.cat([emb_shared_omics1, emb_shared_omics2], dim=0)
        emb_private_for_disc = torch.cat([emb_private_omics1, emb_private_omics2], dim=0)
        logits_shared_modality = self.disc_shared(grad_reverse(emb_shared_for_disc))
        logits_private_modality = self.disc_private(emb_private_for_disc)

        if self.common_fusion_mode == 'concat':
            emb_shared_combined = self.shared_concat_adapter(
                torch.cat([emb_shared_omics1, emb_shared_omics2], dim=1)
            )
            alpha_shared = torch.full(
                (emb_shared_combined.shape[0], 2),
                0.5,
                device=emb_shared_combined.device,
            )
        else:
            emb_shared_combined, alpha_shared = self.shared_mamba_fusion(
                emb_shared_omics1, emb_shared_omics2
            )
        if self.private_fusion_mode == 'attention':
            emb_private_combined, alpha_private = self.atten_private(emb_private_omics1, emb_private_omics2)
        else:
            emb_private_combined = self.private_concat_adapter(
                torch.cat([emb_private_omics1, emb_private_omics2], dim=1)
            )
            alpha_private = torch.full(
                (emb_private_combined.shape[0], 2),
                0.5,
                device=emb_private_combined.device,
            )

        if self.final_fusion_mode == 'residual':
            gate_residual = torch.sigmoid(self.fusion_gate(torch.cat([emb_shared_combined, emb_private_combined], dim=1)))
            residual_delta = self.residual_adapter(emb_private_combined - emb_shared_combined)
            residual_scale = 0.2
            emb_latent_combined = emb_shared_combined + residual_scale * gate_residual * residual_delta
            emb_latent_for_decoder = emb_latent_combined
            alpha_fusion = torch.cat([1 - residual_scale * gate_residual, residual_scale * gate_residual], dim=1)
        elif self.final_fusion_mode == 'concat':
            gate_residual = torch.full(
                (emb_shared_combined.shape[0], 1),
                0.5,
                device=emb_shared_combined.device
            )
            emb_latent_combined = torch.cat([emb_shared_combined, emb_private_combined], dim=1)
            emb_latent_for_decoder = self.concat_fusion_adapter(emb_latent_combined)
            alpha_fusion = torch.cat([gate_residual, gate_residual], dim=1)
        elif self.final_fusion_mode == 'shared':
            gate_residual = torch.zeros((emb_shared_combined.shape[0], 1), device=emb_shared_combined.device)
            emb_latent_combined = emb_shared_combined
            emb_latent_for_decoder = emb_latent_combined
            alpha_fusion = torch.cat([torch.ones_like(gate_residual), gate_residual], dim=1)
        else:
            raise ValueError(f"Unsupported final_fusion_mode: {self.final_fusion_mode}")

        emb_recon_latent_omics1 = self.recon_fusion_omics1(
            torch.cat([emb_shared_combined, emb_private_omics1], dim=1)
        )
        emb_recon_latent_omics2 = self.recon_fusion_omics2(
            torch.cat([emb_shared_combined, emb_private_omics2], dim=1)
        )
        emb_recon_omics1 = self.decoder_omics1(emb_recon_latent_omics1, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_recon_latent_omics2, adj_spatial_omics2)
        emb_aux_recon_omics1 = self.decoder_omics1(emb_latent_for_decoder, adj_spatial_omics1)
        emb_aux_recon_omics2 = self.decoder_omics2(emb_latent_for_decoder, adj_spatial_omics2)

        emb_latent_omics1_across_recon = self.encoder_omics2(
            self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2),
            adj_spatial_omics2
        )
        emb_latent_omics2_across_recon = self.encoder_omics1(
            self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1),
            adj_spatial_omics1
        )

        results = {'emb_latent_omics1': emb_latent_omics1,
                   'emb_latent_omics2': emb_latent_omics2,
                   'emb_proj_omics1': emb_shared_omics1,
                   'emb_proj_omics2': emb_shared_omics2,
                   'emb_latent_combined': emb_latent_combined,
                   'emb_recon_omics1': emb_recon_omics1,
                   'emb_recon_omics2': emb_recon_omics2,
                   'emb_aux_recon_omics1': emb_aux_recon_omics1,
                   'emb_aux_recon_omics2': emb_aux_recon_omics2,
                   'emb_latent_omics1_across_recon': emb_latent_omics1_across_recon,
                   'emb_latent_omics2_across_recon': emb_latent_omics2_across_recon,
                   'emb_shared_omics1': emb_shared_omics1,
                   'emb_shared_omics2': emb_shared_omics2,
                   'emb_private_omics1': emb_private_omics1,
                   'emb_private_omics2': emb_private_omics2,
                   'emb_shared_combined': emb_shared_combined,
                   'emb_private_combined': emb_private_combined,
                   'emb_recon_latent_omics1': emb_recon_latent_omics1,
                   'emb_recon_latent_omics2': emb_recon_latent_omics2,
                   'emb_anchor': emb_anchor,
                   'logits_shared_modality': logits_shared_modality,
                   'logits_private_modality': logits_private_modality,
                   'alpha_omics1': alpha_omics1,
                   'alpha_omics2': alpha_omics2,
                   'alpha_anchor_modal': alpha_anchor_modal,
                   'alpha_shared': alpha_shared,
                   'alpha_private': alpha_private,
                   'alpha': alpha_fusion
                   }

        return results


class ModalityAwareMambaFusion(Module):
    """Adaptive fusion of two modality-common representations."""

    SUPPORTED_MODES = ('conv_gate', 'bidirectional_gate', 'adaptive_weighting', 'concat')
    BIDIRECTIONAL_RESIDUAL_SCALE = 0.2

    def __init__(self, dim_feat, dropout=0.0, fusion_mode='conv_gate'):
        super(ModalityAwareMambaFusion, self).__init__()
        self.dim_feat = dim_feat
        if fusion_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported common fusion mode: {fusion_mode}. "
                f"Expected one of {self.SUPPORTED_MODES}."
            )
        self.fusion_mode = fusion_mode

        self.norm = nn.LayerNorm(dim_feat)
        self.shared_in = nn.Linear(dim_feat, dim_feat)
        self.omics1_in = nn.Linear(dim_feat, dim_feat)
        self.omics2_in = nn.Linear(dim_feat, dim_feat)

        self.value_proj = nn.Linear(dim_feat, dim_feat)
        self.gate_proj = nn.Linear(dim_feat, dim_feat)
        self.depthwise = nn.Conv1d(dim_feat, dim_feat, kernel_size=3, padding=1, groups=dim_feat)
        self.state_proj = nn.Linear(dim_feat, dim_feat)

        self.shared_out = nn.Linear(dim_feat, dim_feat)
        self.omics1_out = nn.Linear(dim_feat, dim_feat)
        self.omics2_out = nn.Linear(dim_feat, dim_feat)
        self.score_proj = nn.Linear(dim_feat, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb1, emb2):
        if self.fusion_mode == 'concat':
            raise RuntimeError("concat fusion is handled by Encoder_overall")

        token1 = self.shared_in(emb1) + self.omics1_in(emb1)
        token2 = self.shared_in(emb2) + self.omics2_in(emb2)
        tokens = torch.stack([token1, token2], dim=1)
        tokens_norm = self.norm(tokens)

        if self.fusion_mode == 'conv_gate':
            value = self.value_proj(tokens_norm)
            gate = torch.sigmoid(self.gate_proj(tokens_norm))
            mixed = self.depthwise(value.transpose(1, 2)).transpose(1, 2)
            mixed = torch.tanh(self.state_proj(mixed))
            tokens_mixed = tokens + self.dropout(gate * mixed)
        elif self.fusion_mode == 'bidirectional_gate':
            norm1 = tokens_norm[:, 0, :]
            norm2 = tokens_norm[:, 1, :]

            candidate_to1 = torch.tanh(self.state_proj(self.value_proj(norm2)))
            candidate_to2 = torch.tanh(self.state_proj(self.value_proj(norm1)))
            agreement = norm1 * norm2
            gate_to1 = torch.sigmoid(self.gate_proj(norm1 + agreement))
            gate_to2 = torch.sigmoid(self.gate_proj(norm2 + agreement))

            scale = self.BIDIRECTIONAL_RESIDUAL_SCALE
            mixed1 = token1 + scale * self.dropout(gate_to1 * candidate_to1)
            mixed2 = token2 + scale * self.dropout(gate_to2 * candidate_to2)
            tokens_mixed = torch.stack([mixed1, mixed2], dim=1)
        else:
            # Retain modality-aware transformations and adaptive scoring while
            # removing explicit cross-modal interaction.
            tokens_mixed = tokens

        out1 = self.shared_out(tokens_mixed[:, 0, :]) + self.omics1_out(tokens_mixed[:, 0, :])
        out2 = self.shared_out(tokens_mixed[:, 1, :]) + self.omics2_out(tokens_mixed[:, 1, :])
        out_tokens = torch.stack([out1, out2], dim=1)

        scores = self.score_proj(torch.tanh(out_tokens)).squeeze(-1)
        alpha = F.softmax(scores + 1e-6, dim=1)
        emb_combined = torch.sum(out_tokens * alpha.unsqueeze(-1), dim=1)

        return emb_combined, alpha

class ModalityDiscriminator(Module):
    def __init__(self, in_feat, hidden_feat=64):
        super(ModalityDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, hidden_feat),
            nn.ReLU(),
            nn.Linear(hidden_feat, 2)
        )

    def forward(self, feat):
        return self.net(feat)


class Encoder(Module):
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x


class ResidualGCNEncoder(Module):
    """One-hop GCN with normalization, activation, and residual projection."""

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(ResidualGCNEncoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.res_linear = nn.Linear(self.in_feat, self.out_feat, bias=False)
        self.norm = nn.LayerNorm(self.out_feat)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.res_linear.weight)

    def forward(self, feat, adj):
        feat_drop = F.dropout(feat, self.dropout, training=self.training)
        x = torch.mm(feat_drop, self.weight)
        x = torch.spmm(adj, x)
        x = self.norm(x)
        x = self.act(x)
        x = x + self.res_linear(feat_drop)

        return x


class GraphSAGEEncoder(Module):
    """GraphSAGE-style encoder with separate self and neighbor transformations."""

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(GraphSAGEEncoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.self_linear = nn.Linear(self.in_feat, self.out_feat, bias=False)
        self.neigh_linear = nn.Linear(self.in_feat, self.out_feat, bias=False)
        self.res_linear = nn.Linear(self.in_feat, self.out_feat, bias=False)
        self.norm = nn.LayerNorm(self.out_feat)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.self_linear.weight)
        nn.init.xavier_uniform_(self.neigh_linear.weight)
        nn.init.xavier_uniform_(self.res_linear.weight)

    def forward(self, feat, adj):
        feat_drop = F.dropout(feat, self.dropout, training=self.training)
        neigh = torch.spmm(adj, feat_drop)
        x = self.self_linear(feat_drop) + self.neigh_linear(neigh)
        x = self.norm(x)
        x = self.act(x)
        x = x + self.res_linear(feat_drop)

        return x


class GraphAttentionEncoder(Module):
    """Sparse-neighborhood graph attention encoder."""

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(GraphAttentionEncoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.temperature = out_feat ** 0.5

        self.query = nn.Linear(self.in_feat, self.out_feat, bias=False)
        self.key = nn.Linear(self.in_feat, self.out_feat, bias=False)
        self.value = nn.Linear(self.in_feat, self.out_feat, bias=False)
        self.res_linear = nn.Linear(self.in_feat, self.out_feat, bias=False)
        self.norm = nn.LayerNorm(self.out_feat)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.res_linear.weight)

    def forward(self, feat, adj):
        feat_drop = F.dropout(feat, self.dropout, training=self.training)
        q = self.query(feat_drop)
        k = self.key(feat_drop)
        v = self.value(feat_drop)
        adj_dense = adj.to_dense()
        mask = adj_dense > 0
        scores = torch.mm(q, k.t()) / self.temperature
        scores = scores.masked_fill(~mask, -9e15)
        attn = F.softmax(scores, dim=1)
        attn = F.dropout(attn, self.dropout, training=self.training)
        x = torch.mm(attn, v)
        x = self.norm(x)
        x = self.act(x)
        x = x + self.res_linear(feat_drop)

        return x


class GraphTransformerEncoder(Module):
    """Graph-masked self-attention with a small feed-forward block."""

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(GraphTransformerEncoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.temperature = out_feat ** 0.5

        self.input_linear = nn.Linear(self.in_feat, self.out_feat, bias=False)
        self.query = nn.Linear(self.out_feat, self.out_feat, bias=False)
        self.key = nn.Linear(self.out_feat, self.out_feat, bias=False)
        self.value = nn.Linear(self.out_feat, self.out_feat, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(self.out_feat, self.out_feat * 2),
            nn.ReLU(),
            nn.Linear(self.out_feat * 2, self.out_feat)
        )
        self.norm_attn = nn.LayerNorm(self.out_feat)
        self.norm_ffn = nn.LayerNorm(self.out_feat)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_linear.weight)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, feat, adj):
        h = self.input_linear(F.dropout(feat, self.dropout, training=self.training))
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        adj_dense = adj.to_dense()
        mask = adj_dense > 0
        scores = torch.mm(q, k.t()) / self.temperature
        scores = scores.masked_fill(~mask, -9e15)
        attn = F.softmax(scores, dim=1)
        attn = F.dropout(attn, self.dropout, training=self.training)
        h = self.norm_attn(h + torch.mm(attn, v))
        h = self.norm_ffn(h + self.ffn(h))

        return h


class GraphMambaEncoder(Module):
    """Mamba-inspired gated graph diffusion encoder."""

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(GraphMambaEncoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.input_linear = nn.Linear(self.in_feat, self.out_feat, bias=False)
        self.neigh_linear = nn.Linear(self.in_feat, self.out_feat, bias=False)
        self.gate_linear = nn.Linear(self.out_feat * 2, self.out_feat)
        self.out_linear = nn.Linear(self.out_feat, self.out_feat, bias=False)
        self.norm = nn.LayerNorm(self.out_feat)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_linear.weight)
        nn.init.xavier_uniform_(self.neigh_linear.weight)
        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.zeros_(self.gate_linear.bias)
        nn.init.xavier_uniform_(self.out_linear.weight)

    def forward(self, feat, adj):
        feat_drop = F.dropout(feat, self.dropout, training=self.training)
        self_state = self.input_linear(feat_drop)
        neigh_state = self.neigh_linear(torch.spmm(adj, feat_drop))
        gate = torch.sigmoid(self.gate_linear(torch.cat([self_state, neigh_state], dim=1)))
        x = gate * neigh_state + (1 - gate) * self_state
        x = self.norm(self.out_linear(x) + self_state)
        x = self.act(x)

        return x


class Decoder(Module):
    """Modality-specific feature decoder with graph smoothing."""

    def __init__(self, in_feat, out_feat, hidden_mult=1.0, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.hidden_feat = max(out_feat, int(in_feat * hidden_mult))
        self.dropout = dropout
        self.act = act

        self.net = nn.Sequential(
            nn.Linear(self.in_feat, self.hidden_feat),
            nn.LayerNorm(self.hidden_feat),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_feat, self.out_feat)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, feat, adj):
        x = self.net(feat)
        x = torch.spmm(adj, x)

        return x


class AttentionLayer(Module):
    """Attention layer for fusing two latent views."""

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = [
            torch.unsqueeze(torch.squeeze(emb1), dim=1),
            torch.unsqueeze(torch.squeeze(emb2), dim=1)
        ]
        emb = torch.cat(emb, dim=1)
        v = torch.tanh(torch.matmul(emb, self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(torch.squeeze(vu) + 1e-6, dim=1)
        emb_combined = torch.matmul(torch.transpose(emb, 1, 2), torch.unsqueeze(alpha, -1))

        return torch.squeeze(emb_combined), alpha
