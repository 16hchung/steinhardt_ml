from .a_PCA import compute as PCAcomp, plot as PCAplot
from .b_tSNE import compute as tSNEcomp, plot as tSNEplot

def main():
  print('computing and plotting PCA')
  PCAcomp.main()
  PCAplot.main()
  if tSNEcomp.has_run_already():
    print('plotting tSNE')
    tSNEplot.main()
  else:
    print('Cannot plot tSNE because haven\'t computed data. See 03_vis/b_tSNE/compute_tsne_job.sh for instructions to compute first')

if __name__=='__main__':
  main()
