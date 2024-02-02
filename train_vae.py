import argparse
import torch
import logging
import os
import numpy as np

import torch

from datasets import QM9
from models import SGP_VAE
from egnn_models import get_adj_matrix
from utils import Queue, gradient_clipping



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--exp_name', type=str, default=None, help='specify experiment name for saving outputs')
    parser.add_argument('--property', type=str, default='homo', help='homo | lumo | dipole_moment | multi')
    parser.add_argument('--max_epoch', type=int, default = 50)
    args = parser.parse_args()
    return args
    
def main():
    args = parse_config()
    exp_dir = './outputs/{}/{}'.format(args.exp_name, args.property)
    num_targets = 3 if args.property == 'multi' else 1


    os.makedirs(exp_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler('{}/exp.log'.format(exp_dir)))

    qm9_npz_path = './qm9_dataset.npz'
    npz_file = np.load(qm9_npz_path, allow_pickle = True)
    selfies = npz_file['selfies']
    homos = npz_file['homo'] * 27.2114
    lumos = npz_file['lumo'] * 27.2114
    try:
        dipole_moments = npz_file['mu']
    except:
        dipole_moments = npz_file['dipole_moment']
    positions = npz_file['positions']
    charges = npz_file['charges']

    dataset_npz_path = '{}/original_dataset.npz'.format(exp_dir)
    np.savez(dataset_npz_path, positions = positions, charges = charges, selfies = selfies, homo = homos, lumo = lumos, dipole_moment = dipole_moments)


    dataset = QM9(dataset_npz_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 32,
        shuffle = True,
        num_workers = 4,
        drop_last = False
    )

    vae_model = SGP_VAE(latent_dim = 256, num_target_properties = num_targets, max_selfies_length = dataset.max_selfies_len, num_alphabet_classes = dataset.num_alphabet_classes)
    vae_model = vae_model.cuda()

    opt = torch.optim.AdamW(vae_model.parameters(), lr = 5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = args.max_epoch * len(dataloader), eta_min = 1e-5)

    gradnorm_queue = Queue(max_len = 200)
    gradnorm_queue.add(100)
    
    for epoch_id in range(args.max_epoch):
        for batch_id, data_dic in enumerate(dataloader):
            selfies_gt = data_dic['selfies_tokens'].cuda()      
            x = data_dic['positions'].cuda()
            h = data_dic['one_hots'].cuda()
            node_mask = data_dic['node_mask'].cuda()

            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            #mask diagonal
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0).cuda()
            edge_mask *= diag_mask

            bs, n_nodes, _ = x.shape
            edges_dic = {}
            edges = get_adj_matrix(edges_dic, n_nodes, bs, 'cuda')

            latent, mu = vae_model.encode(selfies_gt, x.float(), h.float(), node_mask, edges, edge_mask)
            selfies_rec, x_rec, h_rec = vae_model.decode(latent, node_mask, edges, edge_mask)

            pred_prop = vae_model.decode_property(latent)
            if pred_prop.shape[-1] == 1:
                loss, loss_dic = vae_model.compute_loss(selfies_rec, selfies_gt, mu, latent, data_dic[args.property].cuda(), pred_prop, x_rec, x, h_rec, h, node_mask)
            else:
                gt_props = torch.stack((data_dic['homo'].cuda(), data_dic['lumo'].cuda(), data_dic['dipole_moment'].cuda()), dim = -1)
                loss, loss_dic = vae_model.compute_loss(selfies_rec, selfies_gt, mu, latent, gt_props, pred_prop, x_rec, x, h_rec, h, node_mask)
            
            if not(torch.isfinite(loss)):
                opt.zero_grad()
                continue
            loss.backward()
            grad_norm = gradient_clipping(vae_model, gradnorm_queue)    

            opt.step()
            opt.zero_grad()
            sch.step()

            if batch_id % 200 == 0:
                str_log = f'Epoch: {epoch_id}, iter: {batch_id}, loss: {round(loss.item(), 4)}, GradNorm: {round(grad_norm.item(), 4)} '
                for key, val in loss_dic.items():
                    str_log += '{}: {} '.format(key, round(val, 4))
                logger.info(str_log)

        torch.save(vae_model.state_dict(), '{}/vae.pt'.format(exp_dir))

    # Post-training latent-property pairs acquisition    
    vae_model.eval()

    latent_seeds = torch.randn((20000, vae_model.latent_dim)).cuda()
    latent_seeds[:4000] = latent_seeds[:4000] * 2

    new_selfies_list = []
    
    with torch.no_grad(): 
            selfie_tokens = torch.argmax(vae_model.decode(latent_seeds), dim = -1)
            logging.info('Num Molecules: {}'.format(len(latent_seeds)))
            for selfies in selfie_tokens:
                selfie_alphabets = []
                for idx in selfies:
                    selfie_alphabets.append(dataset.idx_to_alphabet[idx.item()])
                selfies = ''.join(selfie_alphabets)
                new_selfies_list.append(selfies)
    np.savez('{}/acquired_latents.npz'.format(exp_dir), opt_selfies = new_selfies_list,  latent = latent_seeds.cpu().detach().numpy())

if __name__ == '__main__':
    main()
