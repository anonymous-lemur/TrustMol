import argparse
import torch
import logging
import os
import numpy as np
from glob import glob

from datasets import QM9
from models import SGP_VAE, LatentToProperty

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--exp_name', type=str, default=None, help='specify experiment name for saving outputs')
    parser.add_argument('--property', type=str, default='homo', help='homo | lumo | dipole_moment | multi')
    parser.add_argument('--max_epoch', type=int, default = 1000)
    parser.add_argument('--num_samples', type=int, default = 2000)
    parser.add_argument('--budget', type = int, default = 10)
    args = parser.parse_args()

    return args

def main():
    args = parse_config()
    exp_dir = './outputs/{}/{}'.format(args.exp_name, args.property)
    num_targets = 3 if args.property == 'multi' else 1

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler('{}/mol_design_optimization.log'.format(exp_dir)))
    logger.addHandler(logging.StreamHandler())

    curr_npz_path = '{}/original_dataset.npz'.format(exp_dir)
    dataset = QM9(curr_npz_path)

    vae_model = SGP_VAE(latent_dim = 256, num_target_properties = num_targets, max_selfies_length = dataset.max_selfies_len, num_alphabet_classes = dataset.num_alphabet_classes)
    vae_model = vae_model.cuda()
    vae_model.load_state_dict(torch.load('{}/vae.pt'.format(exp_dir))) 
    vae_model.eval()

    for param in vae_model.parameters():
        param.requires_grad = False

    ensemble_surrogate_model = []
    for path in glob('{}/surrogate*.pt'.format(exp_dir)):
        _, surrogate_idx, act_fn = path.split('/')[-1].split('-')    
        act_fn = act_fn.split('.')[0]    
        surrogate_model = LatentToProperty(act_fn = act_fn, num_targets=num_targets).cuda()
        surrogate_model.load_state_dict(torch.load(path))
        surrogate_model.eval()
        for param in surrogate_model.parameters():
            param.requires_grad = False
        ensemble_surrogate_model.append(surrogate_model)


    if args.property == 'homo':
        targets = torch.linspace(start = -10., end = 0., steps = args.num_samples).cuda()
    elif args.property == 'lumo':
        targets = torch.linspace(start = -4., end = 2., steps = args.num_samples).cuda()
    elif args.property == 'dipole_moment':
        targets = torch.linspace(start = 0, end = 4., steps = args.num_samples).cuda()
    elif args.property == 'multi':
        targets_homo = torch.linspace(start = -8., end = -3., steps = 10).cuda()
        targets_lumo = torch.linspace(start = -3., end = 2., steps = 10).cuda()
        targets_dipole_moment = torch.linspace(start = 0., end = 4., steps = 10).cuda()
        
        targets_homo = targets_homo.view(-1, 1, 1).repeat(1, 10, 10)
        targets_lumo = targets_lumo.view(1, -1, 1).repeat(10, 1, 10)
        targets_dipole_moment = targets_dipole_moment.view(1, 1, -1).repeat(10, 10, 1)
        targets = torch.stack((targets_homo, targets_lumo, targets_dipole_moment), dim = -1).view(-1, 3)

        num_targets = 1000

    for try_id in range(args.budget):    
        latent_seeds = torch.randn((len(targets), vae_model.latent_dim)).cuda() * 2
        latent_seeds.requires_grad = True
        opt = torch.optim.AdamW([latent_seeds], lr = 1e-1)

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = args.max_epoch, eta_min = 1e-3)

        for iter_id in range(args.max_epoch):
            pred_props = []
            for surrogate in ensemble_surrogate_model:
                pred_props.append(surrogate(latent_seeds))
            pred_props = torch.stack(pred_props)
            pred_mean = torch.mean(pred_props, dim = 0)

            epistemic_ua = torch.nn.functional.l1_loss(pred_props, pred_mean.unsqueeze(0).repeat(pred_props.shape[0], 1, 1))
            surrogate_error = torch.nn.functional.l1_loss(pred_mean.squeeze(), targets)

            std, mean = torch.std_mean(latent_seeds)
            latent_mean = torch.mean(latent_seeds)
            latent_std = torch.std(latent_seeds)
            pull_force = latent_mean ** 2 + (max(0, latent_std - 2)) ** 2 # N(0, 2)
            
            loss = surrogate_error + 1 * epistemic_ua + 1 * pull_force

            loss.backward()
            opt.step()
            sch.step()
            opt.zero_grad()

            if iter_id % 100 == 0:
                logger.info('Iter: {}, Surrogate error: {}, Epistemic UA: {}, Mean: {}, Std: {}'.format(iter_id, surrogate_error.item(), epistemic_ua.item(), std.item(), mean.item()))

        with torch.no_grad():
            pred_props = []
            for surrogate in ensemble_surrogate_model:
                pred_props.append(surrogate(latent_seeds))
            pred_props = torch.stack(pred_props)
            pred_mean = torch.mean(pred_props, dim = 0)

            epistemic_ua = torch.nn.functional.l1_loss(pred_props, pred_mean.unsqueeze(0).repeat(pred_props.shape[0], 1, 1), reduction = 'none').mean(0).squeeze()
            surrogate_error = torch.nn.functional.l1_loss(pred_mean.squeeze(), targets, reduction = 'none')
            selfies_tokens = vae_model.decode(latent_seeds)
            selfies_tokens = torch.argmax(selfies_tokens, dim = -1)


            # Convert to Selfies and String
            saved_selfies = []
            os.makedirs('{}/generated_molecules'.format(exp_dir), exist_ok=True)

            for selfies_idx, selfies in enumerate(selfies_tokens):
                selfie_alphabets = []
                for idx in selfies:
                    selfie_alphabets.append(dataset.idx_to_alphabet[idx.item()])
                selfies = ''.join(selfie_alphabets)
                saved_selfies.append(selfies)

                save_path = '{}/generated_molecules/{}-{}.npz'.format(exp_dir, str(try_id), str(selfies_idx).zfill(4))
                np.savez(save_path, opt_selfies = selfies, gt_prop = targets.detach().cpu().numpy()[selfies_idx], surrogate_error = surrogate_error.detach().cpu().numpy()[selfies_idx], epistemic_ua = epistemic_ua.detach().cpu().numpy()[selfies_idx])
        logging.info('Num Molecules: {}'.format(len(saved_selfies)))


if __name__ == '__main__':
    main()
