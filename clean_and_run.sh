#!/bin/bash

#rm data/*/*
rm post_processing/figures/*/*
rm post_processing/figures/*

#python3 01_create_data.py
python3 02_train_svm_pca.py
python3 03_visualization.py
python3 04_test.py

cd post_processing

python3 01_plot_Q_vs_Q.py
python3 02_plot_tsne_and_pca.py
#python3 03_plot_distortion_pca.py
python3 04_plot_test.py
