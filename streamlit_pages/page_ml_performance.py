import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

import itertools
import random

def page_ml_performance_metrics():
    st.header('Machine Learning Performance Metrics')

    st.subheader('Train, Validation and Test Set: Label Frequencies')

    st.info("""
            These bar plots show the distribution of labels/classes (Healthy and Powdery Mildew) 
            for the train, validation and test sets. As you can see there is a balanced
            distribution between all sets.
        """)
    
    def check_txt_path_exists(path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                path_read = f.read()
            st.text(path_read)
        else:
            st.error(f'File not found: {path}')

    def check_image_path_exists(image_path, caption=None):
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=caption, use_column_width=True)
        else:
            st.error(f'Image path not found {image_path}')


    col1, col2, col3 = st.columns(3)
    with col1:
        train_labels = 'outputs/images/train_set_bar_plot.png'
        check_image_path_exists(train_labels, caption='Train Label Distribution')
    with col2:
        val_labels = 'outputs/images/val_set_bar_plot.png'
        check_image_path_exists(val_labels, caption='Validation Label Distribution')
    with col3:
        test_labels = 'outputs/images/test_set_bar_plot.png'
        check_image_path_exists(test_labels, caption='Test Label Distribution')
    st.write('---')

    st.subheader('Best Model Summary from Hyperparameter Tuning')
    st.info(""" Before creating the final model, I used keras tuner to find the 
            best hyperparameters for model optimisation. The results are shown here. 
        """)
    summary_path = 'outputs/logs/tuner_results_summary.txt'
    check_txt_path_exists(summary_path)

    st.write('---')

    st.subheader('Best Model History')
    col4, col5, col6 = st.columns(3)
    with col4:
        st.info(""" After finding the best model, I fitted it using the train and validation set,
                the batch size was set to 64, here are some plots to show the performance.
            """)
        best_model_path = 'outputs/images/best_model_history_plot.png'
        check_image_path_exists(best_model_path, caption='Best Model History Performance')
    with col5:
        st.info('Here is a classification report detailing the performance.')
        best_model_rep = 'outputs/logs/class_rep_for_best_model.txt'
        check_txt_path_exists(best_model_rep)
    with col6:
        st.info('Here is a confusion matrix detailing the performance.')
        best_model_conf_matrix = 'outputs/images/conf_matrx_for_best_model.png'
        check_image_path_exists(best_model_conf_matrix, caption='Best Model Confusion Matrix')

    st.write('---')

    st.subheader('Cross Validation Search')
