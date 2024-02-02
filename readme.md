# TrustMol

### Dependencies
TrustMol requires RDKit and Psi4 for the NFP-based evaluation.
You can install them with Conda,

```
conda create -n TrustMol -c conda-forge rdkit psi4
```

The rest of the dependencies can be installed with

```
pip install -r requirements.txt
```

Download the preprocessed dataset to train the SGP-VAE from the following link and put it in root directory.
https://drive.google.com/file/d/1NBU-GvJNVKqw9N5bGWnl51sxuItG0mfV/view?usp=sharing

### Training TrustMol
First, train the SGP-VAE.
```
python train_vae.py --exp_name TrustMol --property dipole_moment --max_epoch 50
```

Then, perform the latent-property pairs acquisition to create a new dataset of latent-property pairs.
```
python acquire_lp_pairs.py --slurm_id 1 --num_processes 1 --exp_name TrustMol --property dipole_moment
```
`slurm_id` is the process id if you run the script in parallel. If you run x processes in parallel, each process will need to be assigned a `slurm_id` within the range [0, x). 
Note that this step is cpu-intensive. In our experiments, it takes around 2 hours to complete (using 5000 parallel processes).

After the acquisition, train the ensemble surrogate model with the following line.

```
python train_surrogate.py --exp_name TrustMol --property dipole_moment --max_epoch 300
```

Finally, generate the molecules for evaluation with

```
python generate_molecules.py --exp_name TrustMol --property dipole_moment --max_epoch 1000 --num_samples 2000 --budget 10
```

### Evaluation
To evaluate the generated molecules, first calculate the properties of each molecule with the NFP
```
python calculate_properties.py --exp_name TrustMol --property dipole_moment --slurm_id 1 --num_processes 1
```
Note that this step is cpu-intensive. In our experiments, it takes around 2 hours to complete (using 5000 parallel processes).


Then run the following line
```
python evaluate.py --exp_name TrustMol --property dipole_moment --budget 10
```


### Interactive Tool
The web-based interactive tool for TrustMol is available at https://repo012424.streamlit.app/.
In case of server downtime, the tool can be run locally on a Linux device.
Make sure that the python version is at least 3.10, and install the dependencies,

```
pip install streamlit stpyvista
```

If you plan to run the interactive tool on a headless device, run the following lines to install dependencies

```
sudo apt-install libgl1-mesa-glx xvfb
```

Then, on the `demo` directory, run

```
streamlit run TrustMolGUI.py
```
