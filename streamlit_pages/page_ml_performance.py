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
            st.image(image, caption=caption, use_container_width=True)
        else:
            st.error(f'Image path not found {image_path}')

    def check_csv_path_exists(csv_path):
        if os.path.exists(csv_path):
            csv_results_df = pd.read_csv(csv_path)
            csv_results_styled = csv_results_df.style.highlight_max(axis=0)
            st.dataframe(csv_results_styled)
        else:
            st.error(f'CSV file can not be found {csv_path}')

    option = st.radio(
        'Choose a section you would like to view:',
        ('Label Distribution',
        'Best Model Tuning Summary',
        'Best Model History',
        'Cross Validation Results',
        'Final Model Performance')
    )

    if option == 'Label Distribution':
        st.subheader('Train, Validation and Test Set: Label Frequencies')
        st.info("""
            These bar plots show the distribution of labels/classes (Healthy and Powdery Mildew) 
            for the train, validation and test sets. As you can see there is a balanced
            distribution between all sets.
        """)
        train_labels = 'outputs/images/train_set_bar_plot.png'
        check_image_path_exists(train_labels, caption='Train Label Distribution')
        st.write('---')
        val_labels = 'outputs/images/val_set_bar_plot.png'
        check_image_path_exists(val_labels, caption='Validation Label Distribution')
        st.write('---')
        test_labels = 'outputs/images/test_set_bar_plot.png'
        check_image_path_exists(test_labels, caption='Test Label Distribution')
        st.write('---')

    elif option == 'Best Model Tuning Summary':
        st.subheader('Best Model Summary from Hyperparameter Tuning')
        st.info(""" Before creating the final model, I used keras tuner to find the 
                best hyperparameters for model optimisation. The results are shown here. 
                """)
        summary_path = 'outputs/logs/tuner_results_summary.txt'
        check_txt_path_exists(summary_path)
        st.write('---')

    elif option == 'Best Model History':
        st.subheader('Best Model History')
        st.info(""" After finding the best model, I fitted it using the train and validation set,
                the batch size was set to 64, here are some plots to show the performance.
            """)
        best_model_path = 'outputs/images/best_model_history_plot.png'
        check_image_path_exists(best_model_path, caption='Best Model History Performance')
        st.write('---')
        st.info('Here is a classification report detailing the performance.')
        best_model_rep = 'outputs/logs/class_rep_for_best_model.txt'
        check_txt_path_exists(best_model_rep)
        st.write('---')
        st.info('Here is a confusion matrix detailing the performance.')
        best_model_conf_matrix = 'outputs/images/conf_matrx_for_best_model.png'
        check_image_path_exists(best_model_conf_matrix, caption='Best Model Confusion Matrix')
        st.write('---')

    elif option == 'Cross Validation Results':
        st.subheader('Cross Validation Search')
        st.info(""" After finding the best hyper parameters I created a model using them
                as the values for my next model. This model then performed a cross validation check
                using the stratified k-fold method. The results showed that the model was 
                generalising well and there were no signs of fold dependent overfitting.
                Here are the results from the cross validation search, it achieved an average accuracy 
                of 99.37%. 
            """)
        cross_val_csv_path = 'outputs/logs/cross_validation_results.csv'
        check_csv_path_exists(cross_val_csv_path)
        st.write('---')
        st.info('Here is a classification report detailing the performance of the model.')
        cross_val_model_rep = 'outputs/logs/class_report_cross_val.txt'
        check_txt_path_exists(cross_val_model_rep)
        st.write('---')
        st.info('Here is a confusion matrix detailing the performance of the model.')
        cross_val_model_conf = 'outputs/images/cross_val_conf_matrix.png'
        check_image_path_exists(cross_val_model_conf, caption='Cross Validaiton Model Confusion Matrix')
        st.write('---')
    
    elif option == 'Final Model Performance':
        st.subheader('Final Model History')
        st.info(""" After checking performance metrics throughout the final model was created using
                the best hyperparameters along with the knowledge it is not over fitting. The final
                model was trained using a full train dataset which is a combination dataset of the original
                train and validaiton sets.  
            """)
        final_model_csv_path = 'outputs/logs/final_model_history.csv'
        check_csv_path_exists(final_model_csv_path)
        st.write('---')
        final_model_img_path = 'outputs/images/final_model_history_plot.png'
        check_image_path_exists(final_model_img_path)
        st.write('---')
        st.info('Here is a classification report detailing the performance of the final model.')
        final_model_class_rep = 'outputs/logs/final_model_class_report.txt'
        check_txt_path_exists( final_model_class_rep)
        st.write('---')
        st.info('Here is a confusion matrix detailing the performance of the model.')
        final_model_conf_matrix = 'outputs/images/final_model_conf_matrix.png'
        check_image_path_exists(final_model_conf_matrix, caption='Final Model Confusion Matrix')
        st.write('---')