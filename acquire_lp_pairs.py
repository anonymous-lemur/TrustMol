import numpy as np
import os
import psi4
import time

from rdkit import Chem
from rdkit.Chem import AllChem
import selfies as sf
sf.set_semantic_constraints('octet_rule')

def selfies_to_xyz(selfies, max_tries = 10):
    smiles = sf.decoder(selfies)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    is_done = False
    num_tries = 0
    while not(is_done) and (num_tries < max_tries):
        try:
            AllChem.EmbedMolecule(mol, useRandomCoords=True) 
            AllChem.UFFOptimizeMolecule(mol)
            is_done = True
        except:
            time.sleep(0.1)
            num_tries += 1

    if is_done:
        mol_block = Chem.MolToMolBlock(mol)
        lines = mol_block.strip().split('\n')[3:]  
        str_res = ''
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                atom_type, x, y, z = parts[3], parts[0], parts[1], parts[2]
                if atom_type.isalpha():
                    str_res += '{} {} {} {}\n'.format(atom_type, x, y, z)
    else:
        str_res = ''
    return str_res

def hartree2ev(hartree):
    return 27.2114 * hartree

def get_properties(xyz_string, id = 0):
    '''
        return homo lumo and gap in eV
    '''
    psi4.core.set_output_file("./temp_dat/output{}.dat".format(id), False)
    psi4.set_memory('24 GB')
    psi4.set_num_threads(4)
    psi4.set_options({'basis': 'aug-cc-pVTZ', 'scf_type': 'df', 'cc_type': 'df'})

    try:
        xyz_string = '\n' + xyz_string + "units angstrom\n"
        molecule = psi4.geometry(xyz_string)
        scf_e, scf_wfn = psi4.energy("B3LYP/6-311G(2df,p)", molecule=molecule, return_wfn=True)
        homo = hartree2ev(scf_wfn.epsilon_a_subset("AO", "ALL").np[scf_wfn.nalpha() - 1]) # EV
        lumo = hartree2ev(scf_wfn.epsilon_a_subset("AO", "ALL").np[scf_wfn.nalpha()]) # EV
        gap = lumo - homo
        dipole_moment = np.linalg.norm(np.array(scf_wfn.variable("SCF DIPOLE"))).item()
    except:
        homo = None
        lumo = None
        gap = None
        dipole_moment = None

    psi4.core.clean()
    try:
        os.remove('./temp_dat/output{}.dat'.format(id))
    except:
        pass
    time.sleep(1)

    return homo, lumo, gap, dipole_moment

def calculate_properties(slurm_id, num_processes, npz_file, npz_path):
    for i in range(slurm_id - 1, len(npz_file['opt_selfies']), num_processes):
        dir_id = str(i % 5)
        save_path = os.path.join('/'.join(npz_path.split('/')[:-1]), 'acquired_molecules', dir_id, str(i).zfill(5) + '.npz')
        os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
        
        if os.path.exists(save_path):
            continue
        
        if not('opt_xyz' in npz_file.keys()):
            opt_xyz = selfies_to_xyz(npz_file['opt_selfies'][i].item(), max_tries = 20)
        else:
            try:
                opt_xyz = npz_file['opt_xyz'].item()
            except:
                opt_xyz = npz_file['opt_xyz']
    
        opt_xyz = str(opt_xyz)
        homo, lumo, gap, dipole_moment = get_properties(opt_xyz, id = slurm_id)
       
        save_dic = {
            'selfies': npz_file['opt_selfies'][i],
            'latent': npz_file['latent'][i],
            'opt_xyz': opt_xyz,
            'homo': homo,
            'lumo': lumo,
            'dipole_moment': dipole_moment
        }
        np.savez(save_path, **save_dic)


import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slurm_id', type=int, default=1,
                        help='Specify model path')
    parser.add_argument('--num_processes', type = int, default=1)
    parser.add_argument('--exp_name', type = str, default = None)
    parser.add_argument('--property', type = str, default='homo')
    parser.add_argument('--npz_file_path', type = str, default='acquired_latents.npz')
    args = parser.parse_args()
    
    npz_file_path = './outputs/{}/{}/acquired_latents.npz'.format(args.exp_name, args.property)
    npz_file = np.load(npz_file_path, allow_pickle = True)
    os.makedirs('./temp_dat', exist_ok=True)

    calculate_properties(args.slurm_id, args.num_processes, npz_file, npz_file_path)


if __name__ == '__main__':
    
    main()
