import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from .backbone import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing


class _BaseTrainer:
    def __init__(self,
                 data,
                 datatype='SPOTS',
                 device=torch.device('cpu'),
                 random_seed=2022,
                 learning_rate=0.0001,
                 weight_decay=0.00,
                 epochs=None,
                 dim_input=3000,
                 dim_output=64,
                 weight_factors=[1, 5, 1, 1],
                 n_clusters=None,
                 gnn_encoder='gcn',
                 decoder_hidden_mult_omics1=2.0,
                 decoder_hidden_mult_omics2=1.5,
                 use_recon_loss=True,
                 use_correspondence_loss=True,
                 use_disentangle_loss=True,
                 use_aux_recon_loss=True,
                 shared_adv_weight=1.0,
                 private_cls_weight=1.0,
                 shared_pair_weight=0.5,
                 orthogonality_weight=0.1,
                 disentangle_weight=0.03,
                 aux_recon_weight=0.03,
                 warmup_fraction=0.2,
                 use_residual_fusion=True,
                 final_fusion_mode=None,
                 common_fusion_mode='conv_gate',
                 private_fusion_mode='attention',
                 feature_fusion_mode='current'
                 ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.

        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        self.n_clusters = n_clusters
        self.gnn_encoder = gnn_encoder
        self.decoder_hidden_mult_omics1 = decoder_hidden_mult_omics1
        self.decoder_hidden_mult_omics2 = decoder_hidden_mult_omics2
        self.use_recon_loss = use_recon_loss
        self.use_correspondence_loss = use_correspondence_loss
        self.use_disentangle_loss = use_disentangle_loss
        self.use_aux_recon_loss = use_aux_recon_loss
        self.shared_adv_weight = shared_adv_weight
        self.private_cls_weight = private_cls_weight
        self.shared_pair_weight = shared_pair_weight
        self.orthogonality_weight = orthogonality_weight
        self.disentangle_weight = disentangle_weight
        self.aux_recon_weight = aux_recon_weight
        self.warmup_fraction = warmup_fraction
        self.use_residual_fusion = use_residual_fusion
        self.final_fusion_mode = final_fusion_mode or ('residual' if use_residual_fusion else 'shared')
        self.common_fusion_mode = common_fusion_mode
        self.private_fusion_mode = private_fusion_mode
        self.feature_fusion_mode = feature_fusion_mode

        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(
            self.adata_omics1,
            self.adata_omics2,
            feature_fusion_mode=self.feature_fusion_mode,
        )
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        self.adj_multi_level = self.adj['adj_multi_level'].to(self.device)
        if self.feature_fusion_mode == 'modality_specific':
            self.adj_multi_level_for_model = None
            self.adj_context_omics1 = self.adj_feature_omics1
            self.adj_context_omics2 = self.adj_feature_omics2
        else:
            self.adj_multi_level_for_model = self.adj_multi_level
            self.adj_context_omics1 = None
            self.adj_context_omics2 = None
        self.multi_level_graph_weight = self.adj.get('multi_level_graph_weight', None)
        self.multi_level_graph_diagnostics = self.adj.get('multi_level_graph_diagnostics', None)

        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        self.spatial_order = self._build_spatial_order()

        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs

        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        if self.datatype == 'SPOTS':
            if self.epochs is None:
                self.epochs = 600
            self.weight_factors = [1, 5, 1, 1]

        elif self.datatype == 'Stereo-CITE-seq':
            if self.epochs is None:
                self.epochs = 1500
            self.weight_factors = [1, 10, 1, 10]

        elif self.datatype == '10x':
            if self.epochs is None:
                self.epochs = 350
            self.weight_factors = [1, 1, 1, 1]

        elif self.datatype == 'MISAR-seq':
            if self.epochs is None:
                self.epochs = 350
            self.weight_factors = [10, 1, 1, 1]

        elif self.datatype == 'Spatial-epigenome-transcriptome':
            if self.epochs is None:
                self.epochs = 1600
            self.weight_factors = [1, 5, 1, 1]

    def _build_spatial_order(self):
        if 'spatial' not in self.adata_omics1.obsm:
            return None
        coords = np.asarray(self.adata_omics1.obsm['spatial'], dtype=np.float32)
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        return torch.tensor(order, dtype=torch.long, device=self.device)

    def _adversarial_shared_specific_loss(self, results):
        shared_omics1 = F.normalize(results['emb_shared_omics1'], p=2, eps=1e-12, dim=1)
        shared_omics2 = F.normalize(results['emb_shared_omics2'], p=2, eps=1e-12, dim=1)
        private_omics1 = F.normalize(results['emb_private_omics1'], p=2, eps=1e-12, dim=1)
        private_omics2 = F.normalize(results['emb_private_omics2'], p=2, eps=1e-12, dim=1)
        modality_labels = torch.cat([
            torch.zeros(shared_omics1.shape[0], dtype=torch.long, device=shared_omics1.device),
            torch.ones(shared_omics2.shape[0], dtype=torch.long, device=shared_omics2.device)
        ], dim=0)

        loss_shared_pair = (1 - torch.sum(shared_omics1 * shared_omics2, dim=1)).mean()
        loss_orth_omics1 = torch.sum(shared_omics1 * private_omics1, dim=1).pow(2).mean()
        loss_orth_omics2 = torch.sum(shared_omics2 * private_omics2, dim=1).pow(2).mean()
        loss_shared_adv = F.cross_entropy(results['logits_shared_modality'], modality_labels)
        loss_private_cls = F.cross_entropy(results['logits_private_modality'], modality_labels)
        loss_orth = loss_orth_omics1 + loss_orth_omics2
        self.loss_disentangle_components = {
            'shared_adv': loss_shared_adv,
            'private_cls': loss_private_cls,
            'shared_pair': loss_shared_pair,
            'orthogonality': loss_orth,
        }
        return (
            self.shared_adv_weight * loss_shared_adv
            + self.private_cls_weight * loss_private_cls
            + self.shared_pair_weight * loss_shared_pair
            + self.orthogonality_weight * loss_orth
        )

    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2,
                                     encoder_type=self.gnn_encoder,
                                     decoder_hidden_mult_omics1=self.decoder_hidden_mult_omics1,
                                      decoder_hidden_mult_omics2=self.decoder_hidden_mult_omics2,
                                      use_residual_fusion=self.use_residual_fusion,
                                      final_fusion_mode=self.final_fusion_mode,
                                      common_fusion_mode=self.common_fusion_mode,
                                      private_fusion_mode=self.private_fusion_mode).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), self.learning_rate,
                                          weight_decay=self.weight_decay)
        self.dynamic_loss_weights = torch.FloatTensor(self.weight_factors).to(self.device)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2,
                                 self.adj_multi_level, spatial_order=self.spatial_order)

            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(
                self.features_omics1, results['emb_recon_omics1'], reduction='sum'
            ) / self.features_omics1.numel()
            self.loss_recon_omics2 = F.mse_loss(
                self.features_omics2, results['emb_recon_omics2'], reduction='sum'
            ) / self.features_omics2.numel()
            self.loss_aux_recon_omics1 = F.mse_loss(
                self.features_omics1, results['emb_aux_recon_omics1'], reduction='sum'
            ) / self.features_omics1.numel()
            self.loss_aux_recon_omics2 = F.mse_loss(
                self.features_omics2, results['emb_aux_recon_omics2'], reduction='sum'
            ) / self.features_omics2.numel()
            self.loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
            self.loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])

            loss = torch.tensor(0.0, device=self.device)
            loss_recon_group = self.weight_factors[0] * self.loss_recon_omics1 + \
                               self.weight_factors[1] * self.loss_recon_omics2
            if self.use_recon_loss:
                loss = loss + loss_recon_group
            loss_corr_group = self.weight_factors[2] * self.loss_corr_omics1 + \
                              self.weight_factors[3] * self.loss_corr_omics2
            if self.use_correspondence_loss:
                loss = loss + loss_corr_group
            loss_cluster_group = torch.tensor(0.0, device=self.device)
            loss_disentangle_group = torch.tensor(0.0, device=self.device)
            loss_aux_recon_group = torch.tensor(0.0, device=self.device)
            self.loss_shared_specific = torch.tensor(0.0, device=self.device)
            self.loss_disentangle_components = {}
            if epoch >= int(self.warmup_fraction * self.epochs):
                if self.use_disentangle_loss:
                    self.loss_shared_specific = self._adversarial_shared_specific_loss(results)
                    loss_disentangle_group = self.disentangle_weight * self.loss_shared_specific
                    loss = loss + loss_disentangle_group
                loss_aux_recon_group = self.aux_recon_weight * (
                    self.loss_aux_recon_omics1 + self.loss_aux_recon_omics2
                )
                if self.use_aux_recon_loss:
                    loss = loss + loss_aux_recon_group

            if epoch % 50 == 0 or epoch == self.epochs - 1:
                alpha_mean = results['alpha'].detach().mean(dim=0).cpu().numpy()
                alpha_shared_mean = results['alpha_shared'].detach().mean(dim=0).cpu().numpy()
                tqdm.write(
                    "Epoch [{}/{}] | Total: {:.4f} | "
                    "Recon: {:.4f} (R1 {:.4f}, R2 {:.4f}) | "
                    "Corr: {:.4f} (C1 {:.4f}, C2 {:.4f}) | "
                    "Disent: {:.4f} raw {:.4f} | "
                    "AuxRecon: {:.4f} (A1 {:.4f}, A2 {:.4f}) | "
                    "alpha [{:.4f}, {:.4f}] shared [{:.4f}, {:.4f}]".format(
                        epoch,
                        self.epochs,
                        float(loss.detach().cpu()),
                        float(loss_recon_group.detach().cpu()),
                        float(self.loss_recon_omics1.detach().cpu()),
                        float(self.loss_recon_omics2.detach().cpu()),
                        float(loss_corr_group.detach().cpu()),
                        float(self.loss_corr_omics1.detach().cpu()),
                        float(self.loss_corr_omics2.detach().cpu()),
                        float(loss_disentangle_group.detach().cpu()),
                        float(self.loss_shared_specific.detach().cpu()),
                        float(loss_aux_recon_group.detach().cpu()),
                        float(self.loss_aux_recon_omics1.detach().cpu()),
                        float(self.loss_aux_recon_omics2.detach().cpu()),
                        float(alpha_mean[0]),
                        float(alpha_mean[1]),
                        float(alpha_shared_mean[0]),
                        float(alpha_shared_mean[1]),
                    )
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Model training finished!\n")

        with torch.no_grad():
            self.model.eval()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2,
                                 self.adj_multi_level, spatial_order=self.spatial_order)

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        emb_proj_align = F.normalize(results['emb_proj_omics1'] + results['emb_proj_omics2'], p=2, eps=1e-12, dim=1)
        emb_shared = F.normalize(results['emb_shared_combined'], p=2, eps=1e-12, dim=1)
        emb_private = F.normalize(results['emb_private_combined'], p=2, eps=1e-12, dim=1)
        emb_shared_omics1 = F.normalize(results['emb_shared_omics1'], p=2, eps=1e-12, dim=1)
        emb_shared_omics2 = F.normalize(results['emb_shared_omics2'], p=2, eps=1e-12, dim=1)
        emb_private_omics1 = F.normalize(results['emb_private_omics1'], p=2, eps=1e-12, dim=1)
        emb_private_omics2 = F.normalize(results['emb_private_omics2'], p=2, eps=1e-12, dim=1)

        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'emb_proj_align': emb_proj_align.detach().cpu().numpy(),
                  'SpaMCSRL': emb_combined.detach().cpu().numpy(),
                  'emb_shared_combined': emb_shared.detach().cpu().numpy(),
                  'emb_private_combined': emb_private.detach().cpu().numpy(),
                  'emb_shared_omics1': emb_shared_omics1.detach().cpu().numpy(),
                  'emb_shared_omics2': emb_shared_omics2.detach().cpu().numpy(),
                  'emb_private_omics1': emb_private_omics1.detach().cpu().numpy(),
                  'emb_private_omics2': emb_private_omics2.detach().cpu().numpy(),
                  'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy(),
                  'alpha_molecular': results['alpha_anchor_modal'].detach().cpu().numpy(),
                  'gate_spatial': results['alpha'][:, 1:2].detach().cpu().numpy(),
                  'alpha_anchor_modal': results['alpha_anchor_modal'].detach().cpu().numpy(),
                  'alpha_shared': results['alpha_shared'].detach().cpu().numpy(),
                  'alpha_private': results['alpha_private'].detach().cpu().numpy(),
                  'dynamic_loss_weights': self.dynamic_loss_weights.detach().cpu().numpy(),
                  'loss_corr_omics1': float(self.loss_corr_omics1.detach().cpu()),
                  'loss_corr_omics2': float(self.loss_corr_omics2.detach().cpu()),
                  'loss_aux_recon_omics1': float(self.loss_aux_recon_omics1.detach().cpu()),
                  'loss_aux_recon_omics2': float(self.loss_aux_recon_omics2.detach().cpu()),
                  'loss_shared_specific': float(self.loss_shared_specific.detach().cpu()),
                  'use_recon_loss': self.use_recon_loss,
                  'use_correspondence_loss': self.use_correspondence_loss,
                  'use_disentangle_loss': self.use_disentangle_loss,
                  'use_aux_recon_loss': self.use_aux_recon_loss,
                  'use_residual_fusion': self.use_residual_fusion,
                  'final_fusion_mode': self.final_fusion_mode,
                  'common_fusion_mode': self.common_fusion_mode,
                  'private_fusion_mode': self.private_fusion_mode}
        output['feature_fusion_mode'] = self.feature_fusion_mode
        if self.multi_level_graph_diagnostics is not None:
            output['multi_level_graph_diagnostics'] = self.multi_level_graph_diagnostics
        for name, value in self.loss_disentangle_components.items():
            output[f'loss_{name}'] = float(value.detach().cpu())
        if self.multi_level_graph_weight is not None:
            output['multi_level_graph_weight'] = self.multi_level_graph_weight

        return output
