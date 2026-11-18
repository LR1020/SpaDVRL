import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base_trainer import _BaseTrainer

from .model import _SpaDVRLCoreModel, SpaDVRLModel


class _TrainSpaDVRLCore(_BaseTrainer):
    """Trainer for the formal SpaDVRL configuration.

    Fixed design:
      - no cross-modal correspondence loss;
      - no auxiliary reconstruction loss;
      - raw direct concatenation replaces residual final fusion;
      - only shared-adversarial and private-classification disentanglement;
      - effective weights are 1.0 and 0.3, respectively.
    """

    SHARED_ADV_WEIGHT = 1.0
    PRIVATE_CLS_WEIGHT = 0.3
    DISENTANGLE_WEIGHT = 1.0
    MODEL_VERSION = "SpaDVRL_adaptive_dual_view_representation_learning"
    MODEL_CLASS = _SpaDVRLCoreModel

    def __init__(self, *args, **kwargs):
        locked = {
            "use_correspondence_loss": False,
            "use_aux_recon_loss": False,
            "use_residual_fusion": False,
            "final_fusion_mode": "concat",
            "use_disentangle_loss": True,
            "shared_adv_weight": self.SHARED_ADV_WEIGHT,
            "private_cls_weight": self.PRIVATE_CLS_WEIGHT,
            "shared_pair_weight": 0.0,
            "orthogonality_weight": 0.0,
            "disentangle_weight": self.DISENTANGLE_WEIGHT,
        }
        for name, expected in locked.items():
            if name in kwargs and kwargs[name] != expected:
                raise ValueError(
                    f"{name} is locked to {expected!r} in TrainSpaDVRL."
                )
        kwargs.update(locked)
        super().__init__(*args, **kwargs)

    def _latest_disentangle_loss(self, results):
        shared_omics1 = results["emb_shared_omics1"]
        shared_omics2 = results["emb_shared_omics2"]
        modality_labels = torch.cat(
            [
                torch.zeros(
                    shared_omics1.shape[0],
                    dtype=torch.long,
                    device=shared_omics1.device,
                ),
                torch.ones(
                    shared_omics2.shape[0],
                    dtype=torch.long,
                    device=shared_omics2.device,
                ),
            ],
            dim=0,
        )
        loss_shared_adv = F.cross_entropy(
            results["logits_shared_modality"], modality_labels
        )
        loss_private_cls = F.cross_entropy(
            results["logits_private_modality"], modality_labels
        )
        self.loss_disentangle_components = {
            "shared_adv": loss_shared_adv,
            "private_cls": loss_private_cls,
        }
        return (
            self.SHARED_ADV_WEIGHT * loss_shared_adv
            + self.PRIVATE_CLS_WEIGHT * loss_private_cls
        )

    def train(self):
        self.model = self.MODEL_CLASS(
            self.dim_input1,
            self.dim_output1,
            self.dim_input2,
            self.dim_output2,
            encoder_type=self.gnn_encoder,
            decoder_hidden_mult_omics1=self.decoder_hidden_mult_omics1,
            decoder_hidden_mult_omics2=self.decoder_hidden_mult_omics2,
            common_fusion_mode=self.common_fusion_mode,
            private_fusion_mode=self.private_fusion_mode,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.dynamic_loss_weights = torch.FloatTensor(self.weight_factors).to(
            self.device
        )

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(
                self.features_omics1,
                self.features_omics2,
                self.adj_spatial_omics1,
                self.adj_feature_omics1,
                self.adj_spatial_omics2,
                self.adj_feature_omics2,
                self.adj_multi_level_for_model,
                adj_context_omics1=self.adj_context_omics1,
                adj_context_omics2=self.adj_context_omics2,
                spatial_order=self.spatial_order,
            )
            self.loss_recon_omics1 = F.mse_loss(
                self.features_omics1,
                results["emb_recon_omics1"],
                reduction="sum",
            ) / self.features_omics1.numel()
            self.loss_recon_omics2 = F.mse_loss(
                self.features_omics2,
                results["emb_recon_omics2"],
                reduction="sum",
            ) / self.features_omics2.numel()
            loss_recon_group = (
                self.weight_factors[0] * self.loss_recon_omics1
                + self.weight_factors[1] * self.loss_recon_omics2
            )
            loss = loss_recon_group
            loss_disentangle_group = torch.tensor(0.0, device=self.device)
            self.loss_shared_specific = torch.tensor(0.0, device=self.device)
            self.loss_disentangle_components = {}
            if epoch >= int(self.warmup_fraction * self.epochs):
                self.loss_shared_specific = self._latest_disentangle_loss(results)
                loss_disentangle_group = (
                    self.DISENTANGLE_WEIGHT * self.loss_shared_specific
                )
                loss = loss + loss_disentangle_group

            if epoch % 50 == 0 or epoch == self.epochs - 1:
                tqdm.write(
                    "Epoch [{}/{}] | Total: {:.4f} | Recon: {:.4f} "
                    "(R1 {:.4f}, R2 {:.4f}) | Disent: {:.4f} raw {:.4f}".format(
                        epoch,
                        self.epochs,
                        float(loss.detach().cpu()),
                        float(loss_recon_group.detach().cpu()),
                        float(self.loss_recon_omics1.detach().cpu()),
                        float(self.loss_recon_omics2.detach().cpu()),
                        float(loss_disentangle_group.detach().cpu()),
                        float(self.loss_shared_specific.detach().cpu()),
                    )
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("SpaDVRL training finished!\n")
        with torch.no_grad():
            self.model.eval()
            results = self.model(
                self.features_omics1,
                self.features_omics2,
                self.adj_spatial_omics1,
                self.adj_feature_omics1,
                self.adj_spatial_omics2,
                self.adj_feature_omics2,
                self.adj_multi_level_for_model,
                adj_context_omics1=self.adj_context_omics1,
                adj_context_omics2=self.adj_context_omics2,
                spatial_order=self.spatial_order,
            )

        normalized = {
            "emb_latent_omics1": F.normalize(
                results["emb_latent_omics1"], p=2, eps=1e-12, dim=1
            ),
            "emb_latent_omics2": F.normalize(
                results["emb_latent_omics2"], p=2, eps=1e-12, dim=1
            ),
            "emb_proj_align": F.normalize(
                results["emb_proj_omics1"] + results["emb_proj_omics2"],
                p=2,
                eps=1e-12,
                dim=1,
            ),
            "SpaDVRL": F.normalize(
                results["emb_latent_combined"], p=2, eps=1e-12, dim=1
            ),
            "emb_shared_combined": F.normalize(
                results["emb_shared_combined"], p=2, eps=1e-12, dim=1
            ),
            "emb_private_combined": F.normalize(
                results["emb_private_combined"], p=2, eps=1e-12, dim=1
            ),
            "emb_shared_omics1": F.normalize(
                results["emb_shared_omics1"], p=2, eps=1e-12, dim=1
            ),
            "emb_shared_omics2": F.normalize(
                results["emb_shared_omics2"], p=2, eps=1e-12, dim=1
            ),
            "emb_private_omics1": F.normalize(
                results["emb_private_omics1"], p=2, eps=1e-12, dim=1
            ),
            "emb_private_omics2": F.normalize(
                results["emb_private_omics2"], p=2, eps=1e-12, dim=1
            ),
        }
        output = {
            name: value.detach().cpu().numpy() for name, value in normalized.items()
        }
        output.update(
            {
                "alpha_omics1": results["alpha_omics1"].detach().cpu().numpy(),
                "alpha_omics2": results["alpha_omics2"].detach().cpu().numpy(),
                "alpha": results["alpha"].detach().cpu().numpy(),
                "alpha_molecular": results["alpha_anchor_modal"]
                .detach()
                .cpu()
                .numpy(),
                "gate_spatial": results["alpha"][:, 1:2].detach().cpu().numpy(),
                "alpha_anchor_modal": results["alpha_anchor_modal"]
                .detach()
                .cpu()
                .numpy(),
                "alpha_shared": results["alpha_shared"].detach().cpu().numpy(),
                "alpha_private": results["alpha_private"].detach().cpu().numpy(),
                "dynamic_loss_weights": self.dynamic_loss_weights.detach()
                .cpu()
                .numpy(),
                "loss_shared_specific": float(
                    self.loss_shared_specific.detach().cpu()
                ),
                "loss_shared_adv": float(
                    self.loss_disentangle_components["shared_adv"].detach().cpu()
                ),
                "loss_private_cls": float(
                    self.loss_disentangle_components["private_cls"].detach().cpu()
                ),
                "model_version": self.MODEL_VERSION,
                "shared_adv_weight": self.SHARED_ADV_WEIGHT,
                "private_cls_weight": self.PRIVATE_CLS_WEIGHT,
                "use_correspondence_loss": False,
                "use_aux_recon_loss": False,
                "use_residual_fusion": False,
                "final_fusion_mode": "concat",
                "common_fusion_mode": self.common_fusion_mode,
                "private_fusion_mode": self.private_fusion_mode,
                "feature_fusion_mode": self.feature_fusion_mode,
            }
        )
        if self.multi_level_graph_diagnostics is not None:
            output["multi_level_graph_diagnostics"] = (
                self.multi_level_graph_diagnostics
            )
        if self.multi_level_graph_weight is not None:
            output["multi_level_graph_weight"] = self.multi_level_graph_weight
        return output


class TrainSpaDVRL(_TrainSpaDVRLCore):
    """Trainer for the released SpaDVRL model described in the manuscript."""

    MODEL_VERSION = "SpaDVRL_final_Z_reconstruction"
    MODEL_CLASS = SpaDVRLModel
