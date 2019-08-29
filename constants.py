#from ovito.modifiers import CommonNeighborAnalysisModifier as CNAModifier

structures = [
  ('bcc','lammps_scripts/02_crystals/data/dump_bcc_{}.dat',  3),#int(CNAModifier.Type.BCC)),
  ('fcc','lammps_scripts/02_crystals/data/dump_fcc_{}.dat',  1),#int(CNAModifier.Type.FCC)),
  ('hcp','lammps_scripts/02_crystals/data/dump_hcp_{}.dat',  2),#int(CNAModifier.Type.HCP)),
  ('liq','lammps_scripts/03_liquid/data/dump_liquid_{}.dat', 0)#int(CNAModifier.Type.OTHER))
]

steinhardt_dir = 'data/from_sim'

steinhardt_fname_tmpl = '{}/{}_steinhardt_{}.dat'

train_ts_range = (10000,20000,1000)
test_ts = 20000
