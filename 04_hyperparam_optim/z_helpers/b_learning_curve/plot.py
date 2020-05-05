import matplotlib.pyplot as plt                 
import numpy as np

def plot_learning(data_path, fig_path, add_custom_plot_elements=lambda:None):
  ################################################################################
  # Load and process data.                                                       #
  ################################################################################

  m, acc_train_avg, acc_train_std, acc_valid_avg, acc_valid_std = np.loadtxt(data_path, unpack=True)
  m /= 1000.0

  ################################################################################
  # Plot.                                                                        #
  ################################################################################

  # Start figure.
  fig = plt.figure()
  ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])

  # Plot.
  ax.plot(m, acc_train_avg, 'C3-', lw=2, label='Training set')
  ax.fill_between(m, acc_train_avg-acc_train_std, acc_train_avg+acc_train_std, color='C3', alpha=0.2, lw=0)
  ax.plot(m, acc_valid_avg, 'C0-', lw=2, label='Validation set')
  ax.fill_between(m, acc_valid_avg-acc_valid_std, acc_valid_avg+acc_valid_std, color='C0', alpha=0.2, lw=0)
   
  # Add details.
  ax.set_xlabel(r'Training set size [$\times 10^3$]')
  ax.set_ylabel(r'Accuracy')
  ax.set_xlim(0,50)
  ax.set_ylim(.85,1)
  ax.legend(loc='lower right')
  add_custom_plot_elements()

  # Save figure.
  fig.savefig(fig_path, dpi=300)
  plt.close()

  ################################################################################
