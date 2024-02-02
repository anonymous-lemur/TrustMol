import argparse
import torch
import logging
import os
import numpy as np
import random
from glob import glob
import torch

from datasets import QM9
from models import LatentToProperty
           
    
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--exp_name', type=str, default=None, help='specify experiment name for saving outputs')
    parser.add_argument('--property', type=str, default='homo', help='homo | lumo | dipole_moment | multi')
    parser.add_argument('--max_epoch', type=int, default = 300)
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


    acquired_molecules_dir = '{}/acquired_molecules'.format(exp_dir)
    new_molecule_dirs = os.listdir(acquired_molecules_dir)

    new_selfies = []
    new_homos = []
    new_dipole_moments = []
    new_lumos = []
    new_gaps = []
    new_latents = []
    new_qeds = []
    new_logps = []

    for new_molecule_dir in new_molecule_dirs:
        new_npz_path_list = sorted(glob(os.path.join(acquired_molecules_dir, new_molecule_dir, '*.npz')))

        for new_npz_path in new_npz_path_list:
            new_npz_dic = dict(np.load(new_npz_path, allow_pickle=True))
            selfies = new_npz_dic['selfies'].item()
            latent = new_npz_dic['latent']

            if 'homo' in new_npz_dic.keys():                    
                homo = new_npz_dic['homo'].item()
                lumo = new_npz_dic['lumo'].item()
                dipole_moment = new_npz_dic['dipole_moment'].item()
                if isinstance(homo, float):
                    new_selfies.append(selfies)
                    new_homos.append(homo)
                    new_lumos.append(lumo)
                    new_dipole_moments.append(dipole_moment)
                    new_latents.append(latent)           
            

    new_selfies = np.array(new_selfies)
    if 'homo' in new_npz_dic.keys():
        new_homos = np.array(new_homos)
        new_lumos = np.array(new_lumos)
        new_dipole_moments= np.array(new_dipole_moments)

    new_latents = np.array(new_latents)
    npz_dic = dict()
    
    npz_dic['selfies'] = new_selfies 
    if 'homo' in new_npz_dic.keys():       
        npz_dic['homo'] = new_homos
        npz_dic['lumo'] = new_lumos
        npz_dic['dipole_moment'] = new_dipole_moments
    npz_dic['latent'] = new_latents
    logging.info('New Dataset Length: {}'.format(len(npz_dic['selfies'])))
    dataset_npz_path = '{}/calibrated_dataset.npz'.format(exp_dir)
    np.savez(dataset_npz_path, **npz_dic)

    dataset = QM9(dataset_npz_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 32,
        shuffle = True,
        num_workers = 4,
        drop_last = False
    )

    act_fns = ['relu', 'relu', 'hardswish', 'hardswish', 'silu', 'silu', 'softplus', 'softplus', 'leaky', 'leaky']
    ensemble_surrogate_model = []
    opt_list = []
    sch_list = []


    for act_fn in act_fns:
        surrogate_model = LatentToProperty(act_fn = act_fn, num_targets=num_targets).cuda()
        opt = torch.optim.AdamW(surrogate_model.parameters(), lr = 5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = args.max_epoch * len(dataloader), eta_min = 1e-6)
        ensemble_surrogate_model.append(surrogate_model)
        opt_list.append(opt)
        sch_list.append(sch)


    for epoch_id in range(args.max_epoch):
        accumulated_loss = []
        
        for batch_id, data_dic in enumerate(dataloader):
            latent = data_dic['latent'].cuda()
            mean_loss = []

            for surrogate, opt, sch in zip(ensemble_surrogate_model, opt_list, sch_list):
                if random.random() < 0.3:
                    pred_prop = surrogate(latent)
                    if pred_prop.shape[-1] == 1:
                        loss = torch.nn.functional.l1_loss(pred_prop.squeeze(), data_dic[args.property].cuda().squeeze()) 
                    else:
                        gt_props = torch.stack((data_dic['homo'].cuda(), data_dic['lumo'].cuda(), data_dic['dipole_moment'].cuda()), dim = -1)
                        loss = torch.nn.functional.l1_loss(pred_prop.squeeze(), gt_props.squeeze()) 
                    mean_loss.append(loss.item())         
                    accumulated_loss.append(loss.item())
                    if not(torch.isfinite(loss)):
                        opt.zero_grad()
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(surrogate.parameters(), 10.)
                    opt.step()
                    opt.zero_grad()
                
                sch.step()

            if batch_id % 200 == 0:
                str_log = f'Epoch: {epoch_id}, iter: {batch_id}, loss: {round(np.mean(np.array(mean_loss)).item(), 6)}, '
                logger.info(str_log)

            str_log = f'Epoch: {epoch_id}, Accumulated Loss: {round(np.mean(np.array(accumulated_loss)).item(), 6)}, '
            logger.info(str_log)
                
            for surr_id, (act_fn, surrogate) in enumerate(zip(act_fns, ensemble_surrogate_model)):
                torch.save(surrogate.state_dict(), '{}/surrogate-{}-{}.pt'.format(exp_dir, surr_id, act_fn))

if __name__ == '__main__':
    main()
