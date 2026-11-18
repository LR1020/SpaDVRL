import torch

from .backbone import Encoder_overall, grad_reverse


class _SpaDVRLCoreModel(Encoder_overall):
    """Formal SpaDVRL architecture with the retired branches removed.

    The retained modules are initialized through the original constructor before
    inactive residual modules are deleted. This preserves initialization parity
    with the joint-ablation implementation for a fixed random seed.
    """

    def __init__(
        self,
        dim_in_feat_omics1,
        dim_out_feat_omics1,
        dim_in_feat_omics2,
        dim_out_feat_omics2,
        dropout=0.0,
        act=torch.nn.functional.relu,
        encoder_type="gcn",
        decoder_hidden_mult_omics1=2.0,
        decoder_hidden_mult_omics2=1.5,
        common_fusion_mode="conv_gate",
        private_fusion_mode="attention",
    ):
        super().__init__(
            dim_in_feat_omics1,
            dim_out_feat_omics1,
            dim_in_feat_omics2,
            dim_out_feat_omics2,
            dropout=dropout,
            act=act,
            encoder_type=encoder_type,
            decoder_hidden_mult_omics1=decoder_hidden_mult_omics1,
            decoder_hidden_mult_omics2=decoder_hidden_mult_omics2,
            use_residual_fusion=False,
            final_fusion_mode="concat",
            common_fusion_mode=common_fusion_mode,
            private_fusion_mode=private_fusion_mode,
        )

        # These modules are initialized above only to preserve the RNG sequence
        # of the ablation implementation, then removed from the formal model.
        del self.fusion_gate
        del self.residual_adapter
        del self.concat_fusion_adapter
        self.final_fusion_mode = "concat"
        self.use_residual_fusion = False

    def forward(
        self,
        features_omics1,
        features_omics2,
        adj_spatial_omics1,
        adj_feature_omics1,
        adj_spatial_omics2,
        adj_feature_omics2,
        adj_multi_level=None,
        adj_context_omics1=None,
        adj_context_omics2=None,
        spatial_order=None,
    ):
        del spatial_order
        emb_latent_spatial_omics1 = self.encoder_omics1(
            features_omics1, adj_spatial_omics1
        )
        emb_latent_spatial_omics2 = self.encoder_omics2(
            features_omics2, adj_spatial_omics2
        )

        if adj_multi_level is None:
            adj_context_omics1 = (
                adj_feature_omics1
                if adj_context_omics1 is None
                else adj_context_omics1
            )
            adj_context_omics2 = (
                adj_feature_omics2
                if adj_context_omics2 is None
                else adj_context_omics2
            )
        else:
            adj_context_omics1 = adj_multi_level
            adj_context_omics2 = adj_multi_level

        emb_latent_feature_omics1 = self.encoder_omics1(
            features_omics1, adj_context_omics1
        )
        emb_latent_feature_omics2 = self.encoder_omics2(
            features_omics2, adj_context_omics2
        )
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(
            emb_latent_spatial_omics1, emb_latent_feature_omics1
        )
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(
            emb_latent_spatial_omics2, emb_latent_feature_omics2
        )
        emb_anchor, alpha_anchor_modal = self.atten_cross(
            emb_latent_omics1, emb_latent_omics2
        )

        emb_shared_omics1 = self.shared_proj_omics1(emb_latent_omics1)
        emb_shared_omics2 = self.shared_proj_omics2(emb_latent_omics2)
        emb_private_omics1 = self.private_proj_omics1(emb_latent_omics1)
        emb_private_omics2 = self.private_proj_omics2(emb_latent_omics2)
        logits_shared_modality = self.disc_shared(
            grad_reverse(torch.cat([emb_shared_omics1, emb_shared_omics2], dim=0))
        )
        logits_private_modality = self.disc_private(
            torch.cat([emb_private_omics1, emb_private_omics2], dim=0)
        )

        if self.common_fusion_mode == "concat":
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

        if self.private_fusion_mode == "attention":
            emb_private_combined, alpha_private = self.atten_private(
                emb_private_omics1, emb_private_omics2
            )
        else:
            emb_private_combined = self.private_concat_adapter(
                torch.cat([emb_private_omics1, emb_private_omics2], dim=1)
            )
            alpha_private = torch.full(
                (emb_private_combined.shape[0], 2),
                0.5,
                device=emb_private_combined.device,
            )

        emb_latent_combined = torch.cat(
            [emb_shared_combined, emb_private_combined], dim=1
        )
        alpha_fusion = torch.full(
            (emb_latent_combined.shape[0], 2),
            0.5,
            device=emb_latent_combined.device,
        )

        emb_recon_latent_omics1 = self.recon_fusion_omics1(
            torch.cat([emb_shared_combined, emb_private_omics1], dim=1)
        )
        emb_recon_latent_omics2 = self.recon_fusion_omics2(
            torch.cat([emb_shared_combined, emb_private_omics2], dim=1)
        )
        emb_recon_omics1 = self.decoder_omics1(
            emb_recon_latent_omics1, adj_spatial_omics1
        )
        emb_recon_omics2 = self.decoder_omics2(
            emb_recon_latent_omics2, adj_spatial_omics2
        )

        return {
            "emb_latent_omics1": emb_latent_omics1,
            "emb_latent_omics2": emb_latent_omics2,
            "emb_proj_omics1": emb_shared_omics1,
            "emb_proj_omics2": emb_shared_omics2,
            "emb_latent_combined": emb_latent_combined,
            "emb_recon_omics1": emb_recon_omics1,
            "emb_recon_omics2": emb_recon_omics2,
            "emb_shared_omics1": emb_shared_omics1,
            "emb_shared_omics2": emb_shared_omics2,
            "emb_private_omics1": emb_private_omics1,
            "emb_private_omics2": emb_private_omics2,
            "emb_shared_combined": emb_shared_combined,
            "emb_private_combined": emb_private_combined,
            "emb_recon_latent_omics1": emb_recon_latent_omics1,
            "emb_recon_latent_omics2": emb_recon_latent_omics2,
            "emb_anchor": emb_anchor,
            "logits_shared_modality": logits_shared_modality,
            "logits_private_modality": logits_private_modality,
            "alpha_omics1": alpha_omics1,
            "alpha_omics2": alpha_omics2,
            "alpha_anchor_modal": alpha_anchor_modal,
            "alpha_shared": alpha_shared,
            "alpha_private": alpha_private,
            "alpha": alpha_fusion,
        }


class SpaDVRLModel(_SpaDVRLCoreModel):
    """Final model: both modality-specific decoders reconstruct from final Z."""

    def forward(self, *args, **kwargs):
        results = super().forward(*args, **kwargs)
        integrated = results["emb_latent_combined"]
        recon_latent_omics1 = self.recon_fusion_omics1(integrated)
        recon_latent_omics2 = self.recon_fusion_omics2(integrated)
        results["emb_recon_latent_omics1"] = recon_latent_omics1
        results["emb_recon_latent_omics2"] = recon_latent_omics2
        results["emb_recon_omics1"] = self.decoder_omics1(
            recon_latent_omics1, args[2]
        )
        results["emb_recon_omics2"] = self.decoder_omics2(
            recon_latent_omics2, args[4]
        )
        return results
