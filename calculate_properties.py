import numpy as np
import os
from glob import glob
import psi4
import time
import os

from rdkit import Chem
from rdkit.Chem import AllChem
import selfies as sf

import argparse

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--exp_name', type=str, default=None, help='specify experiment name for saving outputs')
    parser.add_argument('--property', type=str, default='homo', help='homo | lumo | dipole_moment | multi')
    parser.add_argument('--slurm_id', type = int, default = 1)
    parser.add_argument('--num_processes', type = int, default = 1)
    args = parser.parse_args()

    return args

def selfies_to_xyz(selfies, max_tries = 10):
    smiles = sf.decoder(selfies)
    # Create a molecule object from the SMILES string
    mol = Chem.MolFromSmiles(smiles)

    # Generate 3D coordinates
    mol = Chem.AddHs(mol)
    is_done = False
    num_tries = 0
    while not(is_done) and (num_tries < max_tries):
        try:
            AllChem.EmbedMolecule(mol, useRandomCoords=True) 
            # Optimize the 3D structure
            AllChem.MMFFOptimizeMolecule(mol)
            is_done = True
        except:
            time.sleep(0.1)
            num_tries += 1

    if is_done:
        # Convert the molecule to a string in the "atom_type x y z" format
        mol_block = Chem.MolToMolBlock(mol)
        # Parse the mol_block to extract and print atom information
        lines = mol_block.strip().split('\n')[3:]  # Skip the header lines
        str_res = ''
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                atom_type, x, y, z = parts[3], parts[0], parts[1], parts[2]
                # if atom_type in valid_atoms:
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
        dipole_moment = np.linalg.norm(np.array(scf_wfn.variable("SCF DIPOLE"))).item()
    except:
        homo = None
        lumo = None
        dipole_moment = None

    psi4.core.clean()
    try:
        os.remove('./temp_dat/output{}.dat'.format(id))
    except:
        pass
    time.sleep(1)

    return homo, lumo, dipole_moment #, isotropic_dipole_polarizability

def calculate_properties(args, slurm_id, num_processes):
    DIR_PATH = './outputs/{}/{}/generated_molecules'.format(args.exp_name, args.property)
    npz_path_list = sorted(glob(os.path.join(DIR_PATH, '*.npz')))

    for i in range(slurm_id - 1, len(npz_path_list), num_processes):
        try:
            npz_path = npz_path_list[i]
            npz_file = np.load(npz_path, allow_pickle=True)
            npz_file = dict(npz_file)
        except:
            continue
        
        if not('opt_xyz' in npz_file.keys()):
            opt_xyz = selfies_to_xyz(npz_file['opt_selfies'].item(), max_tries = 20)
        else:
            try:
                opt_xyz = npz_file['opt_xyz'].item()
            except:
                opt_xyz = npz_file['opt_xyz']
    
        opt_xyz = str(opt_xyz)
        homo, lumo, dipole_moment = get_properties(opt_xyz, id = slurm_id)
        npz_file['homo'] = np.array([homo])
        npz_file['lumo'] = np.array([lumo])
        npz_file['dipole_moment'] = np.array([dipole_moment])
        npz_file['opt_xyz'] = opt_xyz

        np.savez(npz_path, **npz_file)


def main():
    args = parse_config()
    os.makedirs('./temp_dat', exist_ok=True)

    calculate_properties(args, args.slurm_id, args.num_processes)
    
if __name__ == '__main__':
    
    main()
