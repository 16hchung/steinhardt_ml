from 01_compute_features import 01_compute
from 02_clean_features import compute as 02_clean
from 03_visualization import make_plots as 03_vis

if __name__=='__main__':
  01_compute.main()
  02_clean.main()
  03_vis.main()
