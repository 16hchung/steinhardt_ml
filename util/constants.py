from collections import namedtuple
#from ovito.modifiers import CommonNeighborAnalysisModifier as CNAModifier

######### DIRECTORIES ###############

md_path = '00_md/'
raw_feat_path = '01_compute_features/'
clean_feat_path = '02_clean_features/'
vis_path = '03_visualization/'
pca_path  = '{}a_PCA/'.format(vis_path)
tSNE_path = '{}b_tSNE/'.format(vis_path)
vis_figures_path = '{}figures/'.format(vis_path)
hyperparam_optim_path = '04_hyperparam_optim/'
svm_lin_ovr_path = 'a_svm_linear_ovr/'
svm_lin_ovo_path = 'b_svm_linear_ovo/'
svm_rbf_ovo_path = 'c_svm_rbf_ovo/'

######### CRYSTAL STRUCTURES ########

Lattice= namedtuple('CrystalStruct', 'name, sim_dir, y_label, n_neigh, pt_fmt, ps_pt_fmt')

lattices = [
  Lattice(name='bcc', sim_dir='02_crystals/', y_label=3, n_neigh=8 , pt_fmt='C0s', ps_pt_fmt='C0P'),#int(CNAModifier.Type.BCC)),
  Lattice(name='fcc', sim_dir='02_crystals/', y_label=1, n_neigh=12, pt_fmt='C3o', ps_pt_fmt='C3.'),#int(CNAModifier.Type.FCC)),
  Lattice(name='hcp', sim_dir='02_crystals/', y_label=2, n_neigh=12, pt_fmt='C2^', ps_pt_fmt='C2*'),#int(CNAModifier.Type.HCP)),
  #Lattice(name='liq', sim_dir='03_liquid/',   cna_mod_type=0, n_neigh=None)#int(CNAModifier.Type.OTHER))
]

str_to_latt = {
  'bcc': lattices[0],
  'fcc': lattices[1],
  'hcp': lattices[2],
}
