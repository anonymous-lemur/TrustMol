import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import numpy as np
import selfies as sf
    

pv.start_xvfb() # comment this line if you are not using Linux or have a display


bonds1 = {
    'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
    'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
    'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
        'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
    'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
        'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
        'I': 194},
    'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
        'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
        'I': 187}
}

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}

data_dic = {
    'atoms': [['C', 'H', 'H', 'H', 'H'], ['O', 'H', 'H']],
    'coordinates': [[[0.00000,    0.00000,    0.00000], 
                    [0.00000,    0.00000,    1.08900],
                    [1.02672,    0.00000,   -0.36300],
                    [ -0.51336,   -0.88916,   -0.36300],
                    [ -0.51336,    0.88916,   -0.36300]],

                    [[-0.001,  0.363, -0.000],
                    [-0.825, -0.182, -0.000],
                    [0.826, -0.181,  0.000]]
    ]
}

@st.cache_data
def load_precomputed_results(npz_path):
    mols_dic = dict(np.load(npz_path, allow_pickle=True))['mols'].item()
    return mols_dic

npz_path = './precomputed_results.npz'
mols_dic = load_precomputed_results(npz_path)


def b_generate_callback():
    st.runtime.legacy_caching.clear_cache()
    homo_target = s_homo
    lumo_target = s_lumo
    dipole_target = s_dipole

    chosen_idx = 0
    lowest_nfp_error = 1e8
    
    for idx in range(len(mols_dic['selfies'])):
        d_homo = abs(mols_dic['target'][idx]['homo'] - homo_target)
        d_lumo = abs(mols_dic['target'][idx]['lumo'] - lumo_target)
        d_dipole = abs(mols_dic['target'][idx]['dipole_moment'] - dipole_target)

        if (d_homo <= 0.1) and (d_lumo <= 0.1) and (d_dipole <= 0.1):
            mol_homo, mol_lumo, mol_dipole = mols_dic['property'][idx]['homo'], mols_dic['property'][idx]['lumo'], mols_dic['property'][idx]['dipole_moment']

            if mol_homo.item() is not None:

                nfp_error_homo = abs(mol_homo - homo_target)
                nfp_error_lumo = abs(mol_lumo - lumo_target)
                nfp_error_dipole = abs(mol_dipole - dipole_target)

                nfp_error = (nfp_error_homo + nfp_error_lumo + nfp_error_dipole)/3

                if nfp_error < lowest_nfp_error: # will choose the best molecule out of 10 tries
                    lowest_nfp_error = nfp_error
                    chosen_idx = idx
                        
    show_xyz = mols_dic['xyz'][chosen_idx].item()
    show_selfies = mols_dic['selfies'][chosen_idx].item()
    show_property = mols_dic['property'][chosen_idx]

    st.session_state.show_atoms = [line.split(' ')[0] for line in show_xyz.split('\n')[:-1]]
    st.session_state.show_coords = [line.split(' ')[1:] for line in show_xyz.split('\n')[:-1]]
    st.session_state.show_coords = [[float(x), float(y), float(z)] for (x,y,z) in st.session_state.show_coords]

    if st.session_state.show_mol == True:
        del st.session_state.pv_mol
        st.session_state.show_mol = False
        st.session_state.update_mol = True
    else:
        st.session_state.show_mol = True

    st.session_state.show_smiles = sf.decoder(show_selfies)
    st.session_state.show_property = 'HOMO: {} eV -- LUMO: {} eV -- Dipole Moment: {} D'.format(round(show_property['homo'].item(), 3), 
                                                                                                round(show_property['lumo'].item(), 3), 
                                                                                                round(show_property['dipole_moment'].item(), 3))
    


def generate_graph(atoms, coordinates, node_radius=0.4, edge_radius=0.1):
    """
    Visualize a graph with nodes and edges using PyVista.
    
    :param nodes: A numpy array of node positions (x, y, z).
    :param edges: A list of tuples, each representing an edge (start_node, end_node).
    :param node_radius: Radius of the spheres representing nodes.
    :param edge_radius: Radius of the cylinders representing edges.
    """
    nodes = np.array(coordinates)
    edges = []
    node_color = {'H': 'white', 'C': 'grey', 'N': 'blue', 'O': 'red', 'F': 'green'}

    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            dist =  ((nodes[i] - nodes[j])**2).sum()**0.5 # angstrom
            dist = dist * 100 - 10. # picometer
            bond1_length = bonds1.get(atoms[i], {}).get(atoms[j], 0)
            bond2_length = bonds2.get(atoms[i], {}).get(atoms[j], 0)
            bond3_length = bonds3.get(atoms[i], {}).get(atoms[j], 0)


            if dist <= bond1_length or dist <= bond2_length or dist <= bond3_length:
                edges.append((i, j))

    plotter = pv.Plotter()
    
    # Create and add spheres for each node
    for node, atom in zip(nodes, atoms):
        sphere = pv.Sphere(radius=node_radius, center=node)
        plotter.add_mesh(sphere, color=node_color[atom])
    
    # Create and add cylinders for each edge
    for edge in edges:
        start, end = nodes[edge[0]], nodes[edge[1]]
        line = pv.Line(start, end)
        tube = line.tube(radius=edge_radius)
        plotter.add_mesh(tube, color='black')

    

    return plotter    

st.session_state.range_homo = [-8., -3.]
st.session_state.range_lumo = [-3., 2.]
st.session_state.range_dipole = [0., 4.]

if 'show_mol' not in st.session_state:
    st.session_state.show_mol = False
if 'update_mol' not in st.session_state:
    st.session_state.update_mol = False
if 'show_atoms' not in st.session_state:
    st.session_state.show_atoms = data_dic['atoms'][1]
if 'show_coords' not in st.session_state:
    st.session_state.show_coords = data_dic['coordinates'][1]
if 'show_smiles' not in st.session_state:
    st.session_state.show_smiles = ''
if 'show_property' not in st.session_state:
    st.session_state.show_property = ''

st.markdown("<h1 style='text-align: center;'>TrustMol</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Trustworthy Inverse Molecular Design</h5>", unsafe_allow_html=True)


st.divider()
txt_instruction = st.text('Instruction: ')
txt_ins1 = st.text('1. Set the target values for each property using the sliders.')
txt_ins2 = st.text('2. Click the \'generate\' button to generate the molecule.')
txt_ins3 = st.text('The SMILES string and NFP-calculated properties will be shown in text.')
txt_ins4 = st.text('The molecule will be visualized in 3D under the texts.')
txt_ins5 = st.text('In case the molecule is not shown (usually after the first visualization), click the \'generate\' button again.')


st.divider()

s_homo = st.slider("Highest Occupied Molecular Orbital (eV)", key = "s_homo", disabled = False, 
                   min_value = st.session_state.range_homo[0], max_value = st.session_state.range_homo[1], step = 0.5556, value = -8.)
s_lumo = st.slider("Lowest Unoccupied Molecular Orbital (eV)", key = "s_lumo", disabled = False, 
                   min_value = st.session_state.range_lumo[0], max_value = st.session_state.range_lumo[1], step = 0.5556, value = 2.)
s_dipole = st.slider("Dipole Moment (Debeye)", key = "s_dipole", disabled = False, 
                     min_value = st.session_state.range_dipole[0], max_value = st.session_state.range_dipole[1], step = 0.4444, value = 0.)

b_generate = st.button(label = "generate", on_click=b_generate_callback)

st.divider()

txt_smiles = st.text('Generated SMILES: ' + st.session_state.show_smiles)
txt_props = st.text('Molecule\'s properties:' + st.session_state.show_property)
txt_vis = st.text('3D Visualization: ')

container = st.container(height = 400, border=True)
if st.session_state.show_mol:
    with st.spinner('updating molecule...'):
        with container:
            plotter = generate_graph(st.session_state.show_atoms, st.session_state.show_coords)
            
            ## Final touches
            plotter.view_isometric()
            plotter.background_color = 'white'

            ## Send to streamlit
            stpyvista(plotter, key="pv_mol")
