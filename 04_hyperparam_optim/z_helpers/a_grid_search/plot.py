import matplotlib.pyplot as plt                 
import numpy as np

def plot_validation(param_name, data_path, fig_path, add_custom_plot_elements=lambda:None):
  ################################################################################
  # Input parameters and setup.                                                  #
  ################################################################################

  param, acc_train_avg, acc_train_std, acc_valid_avg, acc_valid_std = np.loadtxt(data_path, unpack=True)

  ################################################################################
  # Plot.                                                                        #
  ################################################################################

  # Start figure.
  fig = plt.figure()
  ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])

  # Plot.
  ax.plot(param, acc_train_avg, 'C3-', lw=2, label='Training set')
  ax.fill_between(param, acc_train_avg-acc_train_std, acc_train_avg+acc_train_std, color='C3', alpha=0.2, lw=0)
  ax.plot(param, acc_valid_avg, 'C0-', lw=2, label='Validation set')
  ax.fill_between(param, acc_valid_avg-acc_valid_std, acc_valid_avg+acc_valid_std, color='C0', alpha=0.2, lw=0)
  add_custom_plot_elements()

  # Add details.
  ax.set_xlabel(param_name)
  ax.set_ylabel(r'Accuracy')
  ax.set_xscale('log')
  ax.set_xlim(param.min(),param.max())
  ax.set_ylim(0.95,1)
  ax.legend(loc='lower right')

  # Save figure.
  fig.savefig(fig_path, dpi=300)
  plt.close()

  ################################################################################
