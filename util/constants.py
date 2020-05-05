import numpy as np
from collections import namedtuple

######### DIRECTORIES ###############

md_path = '00_md/'
raw_feat_path = '01_compute_features/'
clean_feat_path = '02_clean_features/'

vis_path = '03_visualization/'
pca_path  = '{}a_PCA/'.format(vis_path)
tSNE_path = '{}b_tSNE/'.format(vis_path)
zscore_path = '{}d_zscores/'.format(vis_path)
vis_figures_path = '{}figures/'.format(vis_path)

hyperparam_optim_path = '04_hyperparam_optim/'
model_exam_path = '06_model_examination/'

# different approaches
svm_lin_ovr_path = 'a_svm_linear_ovr/'
svm_lin_ovo_path = 'b_svm_linear_ovo/'
svm_rbf_ovo_path = 'c_svm_rbf_ovo/'
ocsvm_rbf_path = 'd_ocsvm_rbf/'
all_svm_lin_ovo_path = 'e_all_svm_lin_ovo/'
cat_svm_lin_ovr_path = 'f_cat_svm_lin_ovr/'
cat_svm_rbf_ovo_path = 'g_cat_svm_rbf_ovo/'
cat_svm_lin_ovo_path = 'h_cat_svm_lin_ovo/' 

######### CRYSTAL STRUCTURES ########

Lattice= namedtuple('CrystalStruct', 'name, sim_dir, T_m, low_temp, high_temp, step_temp, dflt_temp, y_label, n_neigh, pt_fmt, ps_pt_fmt')

lattices = [
  Lattice(name='fcc', sim_dir='02_crystals/', T_m=933,  low_temp=100, high_temp=1100, step_temp=100, dflt_temp=900,  y_label=1, n_neigh=12, pt_fmt='ro', ps_pt_fmt='ko'     ),#int(CNAModifier.Type.FCC)),
  Lattice(name='hcp', sim_dir='02_crystals/', T_m=1941, low_temp=100, high_temp=2200, step_temp=100, dflt_temp=1500, y_label=2, n_neigh=12, pt_fmt='go', ps_pt_fmt='#6a0dad'),#int(CNAModifier.Type.HCP)),
  Lattice(name='bcc', sim_dir='02_crystals/', T_m=1811, low_temp=100, high_temp=2000, step_temp=100, dflt_temp=1500, y_label=3, n_neigh=8 , pt_fmt='bo', ps_pt_fmt='#ff4500'),#int(CNAModifier.Type.BCC)),
  Lattice(name='hd',  sim_dir='02_crystals/', T_m=273,  low_temp=20,  high_temp=340,  step_temp=20,  dflt_temp=220,  y_label=4, n_neigh=16, pt_fmt='mo', ps_pt_fmt='#ffb6c1'),#int(CNAModifier.Type.HCP)),
  Lattice(name='cd',  sim_dir='02_crystals/', T_m=1687, low_temp=100, high_temp=2000, step_temp=100, dflt_temp=1500, y_label=5, n_neigh=16, pt_fmt='yo', ps_pt_fmt='#008080'),#int(CNAModifier.Type.HCP)),
  Lattice(name='sc',  sim_dir='02_crystals/', T_m=1074, low_temp=100, high_temp=1200, step_temp=100, dflt_temp=500,  y_label=6, n_neigh=6,  pt_fmt='co', ps_pt_fmt='#800000'),#int(CNAModifier.Type.HCP)),
  #Lattice(name='liq', sim_dir='03_liquid/',   cna_mod_type=0, n_neigh=None)#int(CNAModifier.Type.OTHER))
]

#possible_n_neigh = list(set([l.n_neigh for l in lattices]))
possible_n_neigh = list(range(1, 17))

str_to_latt = {
  'fcc': lattices[0],
  'hcp': lattices[1],
  'bcc': lattices[2],
  'hd':  lattices[3],
  'cd':  lattices[4],
  'sc':  lattices[5],
}

lbl_to_latt = {l.y_label : l for l in lattices}

n_features = 10

method_to_name = {'PTM':'Polyhedral Template Matching', 'CNA':'Common Neighbor Analysis', 'AJA':'Ackland-Jones Analysis', 'VTM':'VoroTop Analysis', 'CPA':'Chill+'}
