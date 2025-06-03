import streamlit as st

def page_hypothesis():
    st.title('Project Hypothesis')

    st.info("""This project explores whether a convolutional neural network (CNN) based binary classification model can 
            accurately classify cherry leaves as either healthy or ones containing powdery mildew. The model will be
            able to achieve a prediction accuracy of at least a 97%.

            **Null Hypothesis**: A CNN model trained on cherry leaf images cannot reliably distinguish between
            healthy leaves and ones containing powdery mildew better than random chance (around '50%' accuracy).

            **Alternative Hypothesis:** A CNN model trained on cherry leaf images can reliably distinguish between
            healthy leaves and ones containing powdery mildew with at least a '97%' accuracy.
        """)