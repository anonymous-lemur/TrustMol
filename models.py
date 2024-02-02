import torch
import torch.nn as nn
import math

from egnn_models import EGNN_Encoder

class LatentToProperty(torch.nn.Module):
    """
    The E(n) Hierarchical VAE Module.
    """
    def __init__(self, act_fn = 'relu', num_targets = 1):
        super().__init__()
        self.latent_dim = 256
        act_fn_dic = {'silu': nn.SiLU(), 'relu': nn.ReLU(), 'gelu': nn.GELU(), 'softplus': nn.Softplus(), 'hardswish': nn.Hardswish(), 'leaky': nn.LeakyReLU()}
        self.act_fn = act_fn_dic[act_fn]
        
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            self.act_fn,
            nn.Linear(512, 512),
            self.act_fn,
            nn.Linear(512, 512),
            self.act_fn,
            nn.Linear(512, 512),
            self.act_fn,
            nn.Linear(512, 512),
            self.act_fn,
            nn.Linear(512, num_targets),
        )

    def forward(self, latent):
        return self.predictor(latent)
          
class SGP_VAE(torch.nn.Module):
    """
    The E(n) Hierarchical VAE Module.
    """
    def __init__(self, latent_dim = 256, num_target_properties = 1, max_selfies_length = 100, num_alphabet_classes = 66):
        super().__init__()

        self.latent_dim = latent_dim
        self.alphabet = None
        self.alphabet_to_idx = None
        self.idx_to_alphabet = None
        self.num_alphabet_classes = num_alphabet_classes
        self.max_selfies_length = max_selfies_length

        
        self.embedding = nn.Linear(self.num_alphabet_classes, 1)

        self.encoder = nn.Sequential(
            nn.Linear(self.max_selfies_length, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, self.latent_dim)
        )

        self.egnn_encoder = EGNN_Encoder(
            in_node_nf = 5, # H C N O F
            context_node_nf = 0, # No context
            out_node_nf = self.latent_dim,
            n_dims = 3, # 3D Coordinates
            hidden_nf = 192,
            n_layers = 3,
            normalization_factor = [1, 8, 1],
        )

        self.mu_fused = nn.Sequential(
            nn.Linear(2 * self.latent_dim + 3, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, 512),
                                     nn.SiLU(),
                                     nn.Linear(512, 512),
                                     nn.SiLU(),
                                     nn.Linear(512, 512),
                                     nn.SiLU(),
                                     nn.Linear(512, 512),
                                     nn.SiLU(),
                                     nn.Linear(512, 512),
                                     nn.SiLU(),
                                     nn.Linear(512, 512),
                                     nn.SiLU(),
                                     nn.Linear(512, self.max_selfies_length * self.num_alphabet_classes))

        
        self.decoder_3d = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 29 * 8), # max_length x (3d + atom_type)
        )

        self.property_decoder = nn.Sequential(nn.Linear(self.latent_dim, 512),
                                     nn.SiLU(),
                                     nn.Linear(512, 512),
                                     nn.SiLU(),
                                     nn.Linear(512, 256),
                                     nn.SiLU(),
                                     nn.Linear(256, 256),
                                     nn.SiLU(),
                                     nn.Linear(256, num_target_properties)
        )

                                     
    
    def compute_loss(self, selfies_rec, selfies_gt, mu, latent, pred_prop = None, gt_prop = None, 
                     x_rec = None, x = None, h_rec = None, h = None, node_mask = None):
        bs, latent_dim = mu.shape
        bs, n_alphabets, n_classes = selfies_rec.shape
        w_tensor = torch.ones((n_classes,)).cuda()
        w_tensor[-1] = 0.05 # 1:20 for [nop] class
        rec_loss = torch.nn.functional.cross_entropy(selfies_rec.permute(0,2,1), selfies_gt, weight = w_tensor)

        # 3D Reconstruction Loss
        if x_rec is not None:
            x_rec = x_rec.view(bs, -1, 3)
            x = x.view(bs, -1, 3)
            x_loss = ((x - x_rec).pow(2).sum(dim = -1) * node_mask).sum() / node_mask.sum()
            # x_loss = torch.nn.functional.mse_loss(x_rec, x)

            h_loss = torch.nn.functional.cross_entropy(h_rec.permute(0,2,1), h.argmax(dim = -1), reduction = 'none')
            h_loss = (h_loss * node_mask).sum() / node_mask.sum()
            rec_loss_3d = x_loss + h_loss


        kl_div = (-0.5 * torch.sum(1 + 0.1 - mu.pow(2) - math.exp(0.1))) / bs
        kl_div = max(torch.zeros((1,)).cuda(), kl_div - 0.1)

        zero_mean = (torch.mean(latent, dim = 0)**2).sum() / latent.shape[-1]
        one_variance = ((torch.std(latent, dim = 0) - 1)**2).sum() / latent.shape[-1]

        loss = rec_loss + 0.01 * kl_div + rec_loss_3d
        loss_dic = dict()
        loss_dic['rec_loss'] = rec_loss.item()
        loss_dic['kl_div'] = kl_div.item()
        loss_dic['zero_mean'] = zero_mean.item()
        loss_dic['one_variance'] = one_variance.item()
        loss_dic['rec_loss_3d'] = rec_loss_3d.item()
        loss_dic['x_loss'] = x_loss.item()
        loss_dic['h_loss'] = h_loss.item()


        if pred_prop is not None:
            prop_mae = torch.nn.functional.l1_loss(pred_prop.squeeze(), gt_prop.squeeze())
            loss += prop_mae
            loss_dic['prop_mae'] = prop_mae.item()
            
        batch_cls_accuracy = (torch.argmax(selfies_rec, dim = -1) == selfies_gt).float().mean() * 100
        loss_dic['batch_cls_accuracy'] = batch_cls_accuracy.item()

        return loss, loss_dic


    def forward(self):
        raise NotImplementedError
    
    def encode(self, selfie_tokens, x, h, node_mask=None, edges = None, edge_mask=None):
        bs, max_len = selfie_tokens.shape
        one_hot_tokens = torch.nn.functional.one_hot(selfie_tokens, num_classes = self.num_alphabet_classes).float()
        one_hot_tokens = self.embedding(one_hot_tokens)
        one_hot_tokens = one_hot_tokens.view(bs, -1)

        mu_selfies = self.encoder(one_hot_tokens)
        xh = torch.cat([x, h], dim = -1)
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.egnn_encoder._forward(xh, node_mask, edge_mask, context = None)
        mu_3d = torch.cat([z_x_mu, z_h_mu], dim = -1).sum(dim = 1)

        mu = self.mu_fused(torch.cat([mu_selfies, mu_3d], dim = -1))

        latent = mu + torch.randn_like(mu)

        return latent, mu
    
    def decode(self, latent, node_mask = None, edges = None, edge_mask = None):
        bs, latent_dim = latent.shape
        selfies_rec = self.decoder(latent)
        selfies_rec = selfies_rec.view(bs, self.max_selfies_length, self.num_alphabet_classes)


        if node_mask is not None:
            xh_rec = self.decoder_3d(latent).view(bs, -1, 8)
            x_rec, h_rec = xh_rec[:, :, :3], xh_rec[:, :, 3:]
            return selfies_rec, x_rec, h_rec
        else:
            return selfies_rec
        
    def decode_property(self, latent):
        return self.property_decoder(latent)

