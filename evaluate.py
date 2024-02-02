import os
from glob import glob
import numpy as np
import logging
import sys
import argparse

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--exp_name', type=str, default=None, help='specify experiment name for saving outputs')
    parser.add_argument('--property', type=str, default='homo', help='homo | lumo | dipole_moment | multi')
    parser.add_argument('--budget', type = int, default = 10)
    args = parser.parse_args()

    return args

def main():
    args = parse_config()

    NUM_TRIES = args.budget
    property_name = args.property 
    DIR_PATH = './outputs/{}/{}/generated_molecules'.format(args.exp_name, args.property)


    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    

    name_txt = DIR_PATH.split('/')[-1]

    npz_path_list = sorted(glob(os.path.join(DIR_PATH, '0*.npz')))
    nfp_error_list = []
    surrogate_error_list = []
    nfp_surrogate_error_gap_list = []

    num_molecules_invalid = 0
    num_molecules_valid = 0
    for npz_path in npz_path_list:
        lowest_nfp_error = 1e8
        at_least_one_valid = False

        for j in range(NUM_TRIES):
            try:
                npz_splitted = npz_path.split('/')
                npz_file_name = '/'.join(npz_splitted[:-1] + ['-'.join([str(j)] + npz_splitted[-1].split('-')[1:])])
                npz_file = np.load(npz_file_name, allow_pickle = True)
                npz_file = dict(npz_file)
                logging.info('%s', npz_file.keys())
            
                opt_prop = npz_file[property_name].item()
                if opt_prop != None:
                    gt_prop = npz_file['gt_prop'].item()
                    at_least_one_valid = True
                    nfp_error = abs(gt_prop - opt_prop)
                    
                    if nfp_error < lowest_nfp_error:
                        lowest_nfp_error = nfp_error
                        try:
                            surrogate_error = npz_file['surrogate_error'].item()
                        except:
                            surrogate_error = 1e8
                        nfp_surrogate_error_gap = abs(lowest_nfp_error - surrogate_error)
                    
            except:
                pass

        if at_least_one_valid:
            num_molecules_valid += 1 
            nfp_error_list.append(lowest_nfp_error)
            surrogate_error_list.append(surrogate_error)
            nfp_surrogate_error_gap_list.append(nfp_surrogate_error_gap)
        else: 
            num_molecules_invalid += 1

    with open('/'.join(DIR_PATH.split('/')[:-1]) + '/{}_eval_complete.txt'.format(name_txt), 'w') as f:
        f.write('NFP MAE: {}\nSurrogate MAE: {}\n NFP-Surrogate Gap MAE: {}\nNum Molecules: {}; \nNum NFP_errors:{} Valid Rate: {}%'.format(np.mean(nfp_error_list).item(),
                                                                            np.mean(surrogate_error_list).item(),
                                                                            np.mean(nfp_surrogate_error_gap_list).item(), 
                                                                            num_molecules_valid + num_molecules_invalid, 
                                                                            len(nfp_error_list),
                                                                            100 * num_molecules_valid / (num_molecules_valid + num_molecules_invalid)))
    
    logging.info('NFP MAE: {}'.format(np.mean(nfp_error_list)))
    logging.info('Surrogate MAE: {}'.format(np.mean(surrogate_error_list)))
    logging.info('NFP-Surrogate Gap MAE: {}'.format(np.mean(nfp_surrogate_error_gap_list)))
    logging.info('Num Molecules: {}; Valid Rate: {}%'.format(num_molecules_valid + num_molecules_invalid, 100 * num_molecules_valid / (num_molecules_valid + num_molecules_invalid)))

if __name__ == '__main__':
    main()
