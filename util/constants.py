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
svm_lin_ovr_path = 'a_svm_linear_ovr/'
svm_lin_ovo_path = 'b_svm_linear_ovo/'
svm_rbf_ovo_path = 'c_svm_rbf_ovo/'
ocsvm_rbf_path = 'd_ocsvm_rbf/'
all_svm_lin_ovo_path = 'e_all_svm_lin_ovo/'
cat_svm_lin_ovo_path = 'f_cat_svm_lin_ovo/'

######### CRYSTAL STRUCTURES ########

Lattice= namedtuple('CrystalStruct', 'name, sim_dir, y_label, n_neigh, pt_fmt, ps_pt_fmt')

lattices = [
  Lattice(name='bcc', sim_dir='02_crystals/', y_label=3, n_neigh=8 , pt_fmt='bs', ps_pt_fmt='cP'),#int(CNAModifier.Type.BCC)),
  Lattice(name='fcc', sim_dir='02_crystals/', y_label=1, n_neigh=12, pt_fmt='ro', ps_pt_fmt='m.'),#int(CNAModifier.Type.FCC)),
  Lattice(name='hcp', sim_dir='02_crystals/', y_label=2, n_neigh=12, pt_fmt='g^', ps_pt_fmt='y*'),#int(CNAModifier.Type.HCP)),
  #Lattice(name='liq', sim_dir='03_liquid/',   cna_mod_type=0, n_neigh=None)#int(CNAModifier.Type.OTHER))
]

possible_n_neigh = list(set([l.n_neigh for l in lattices]))

str_to_latt = {
  'bcc': lattices[0],
  'fcc': lattices[1],
  'hcp': lattices[2],
}

lbl_to_latt = {l.y_label : l for l in lattices}

n_features = 30
