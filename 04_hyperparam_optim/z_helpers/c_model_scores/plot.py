import matplotlib.pyplot as plt                 
import pandas as pd
import numpy as np

def plot_concat_by_temp(data_path, fig_path):
  #fig = plt.figure()
  #ax = fig.add_axes([.15, 100, .8, 2200])
  scores = pd.read_csv(data_path.format('cat_byT_'))
  scores.set_index('temp', inplace=True)
  scores.groupby('latt')['accuracy'].plot(legend=True)
  #ax.set_xlabel('T (K)')
  #ax.set_ylabel('Accuracy')

  plt.savefig(fig_path, dpi=300)
  #plt.close()
