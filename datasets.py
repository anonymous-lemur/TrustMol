import torch
import numpy as np
import torch
import numpy as np
import selfies as sf
import torch
import numpy as np


class QM9(torch.utils.data.Dataset):
    def __init__(self, npz_file_path):
        self.ha_to_ev = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
        self.charges_to_idx = {0: 0, 1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

        self.npz_file = dict(np.load(npz_file_path, allow_pickle=True))       
        self.keys = self.npz_file.keys()

        homo = self.npz_file['homo']
        lumo = self.npz_file['lumo']
        dipole_moment = self.npz_file['dipole_moment']

        if 'latent' in self.npz_file.keys():
            latent = self.npz_file['latent']
        else:
            latent = None

        self.selfies = self.npz_file['selfies']                   

        self.alphabet = ['[O-1]', '[=N+1]', '[H]', '[CH1-1]', '[\\C-1]', '[/C@H1]', '[N]', '[/C@@H1]', '[=N]', 
                         '[C@]', '[Branch1]', '[\\O]', '[N@@H1+1]', '[NH3+1]', '[#C]', '[\\H]', '[=NH2+1]', 
                         '[C-1]', '[/C-1]', '[\\N]', '[=Branch1]', '[\\C@H1]', '[=Branch2]', '[=Ring1]', '[C@@]', 
                         '[=O+1]', '[/O]', '[C]', '[#N]', '[#Branch1]', '[N-1]', '[/NH1+1]', '[Ring2]', 
                         '[\\C@@H1]', '[/N-1]', '[/N]', '[\\O-1]', '[C@H1]', '[#Branch2]', '[/O-1]', '[\\C]', 
                         '[/NH1]', '[NH1+1]', '[N+1]', '[Ring1]', '[NH2+1]', '[=O]', '[/C]', '[-\\Ring1]', '[N@H1+1]', 
                         '[F]', '[C@@H1]', '[O]', '[=NH1+1]', '[=C]', '[NH1]', '[/H]', '[-/Ring1]', '[Branch2]', '[/C@@]', 
                         '[\\NH1]', '[nop]']
         
        
        self.max_selfies_len = max(sf.len_selfies(s) for s in self.selfies)
        self.alphabet_to_idx = {s: i for i, s in enumerate(self.alphabet)}
        self.idx_to_alphabet = {i: s for i, s in enumerate(self.alphabet)}
        self.num_alphabet_classes = len(self.alphabet_to_idx)

        self.selfies_tokens = []
        self.homos = []
        self.lumos = []
        self.dipole_moments = []
        self.latents = []

        for idx, (selfies, h, l, d) in enumerate(zip(self.selfies, homo, lumo, dipole_moment)):
            tokens = []
            try:
                for s in sf.split_selfies(selfies):
                    tokens.append(self.alphabet_to_idx[s])
                tokens += [self.alphabet_to_idx['[nop]'] for _ in range(self.max_selfies_len - len(tokens))]
                self.selfies_tokens.append(tokens)
                self.homos.append(h)
                self.lumos.append(l)
                self.dipole_moments.append(d)

                if latent is not None:
                    self.latents.append(latent[idx])
            except:
                pass
                
        self.selfies_tokens = np.array(self.selfies_tokens)
        self.homos = np.array(self.homos) # EV
        self.lumos = np.array(self.lumos) # EV
        self.dipole_moments = np.array(self.dipole_moments) # EV

        if latent is None:
            self.latents = None
        else:
            self.latents = np.array(self.latents)
        
        try:
            self.positions = np.array(self.npz_file['positions'])
            self.charges = np.array(self.npz_file['charges'])
        except:
            self.positions = None
            self.charges = None
    def __len__(self):
        return len(self.selfies_tokens)

    def __getitem__(self, idx):
        data_dic = {
                'selfies_tokens': torch.from_numpy(self.selfies_tokens[idx]).long(),
                'homo': torch.tensor(self.homos[idx]).float(),   
                'lumo': torch.tensor(self.lumos[idx]).float(),
                'dipole_moment': torch.tensor(self.dipole_moments[idx]).float(),
            }
        
        if self.positions is not None:
            positions = torch.from_numpy(self.positions[idx])
            charges = self.charges[idx]
            node_mask = charges > 0
            one_hots = torch.nn.functional.one_hot(torch.from_numpy(np.array([self.charges_to_idx[c.item()] for c in charges])), num_classes = 5)
            node_mask = torch.from_numpy(node_mask).long()


            data_dic['positions'] = positions,
            data_dic['node_mask'] = node_mask,
            data_dic['one_hots'] = one_hots



        if self.latents is not None:
            data_dic['latent'] = torch.tensor(self.latents[idx]).float()


        return data_dic
