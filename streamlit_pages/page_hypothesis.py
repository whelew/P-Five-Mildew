import streamlit as st

def page_hypothesis():
    st.title('Project Hypothesis')

    st.write(""" The client is facing a challenge where their cherry plantations have been presenting
             leaves with powdery mildew. It is a manual process that takes an employee around 30 minutes in
             each tree to collect a few samples and veryify whether they contain powdery mildew or they are healthy.
             Therefore the client has asked for a ML model that will allow instant detection of powdery mildew within a 
             cherry leaf image.
    """)

    st.info("""This project explores whether a convolutional neural network (CNN) based binary classification model can 
            accurately classify cherry leaves as either healthy or ones containing powdery mildew. The model will be
            able to achieve a prediction accuracy of at least a **97%**.""")

    st.info("""**Null Hypothesis**: A CNN model trained on cherry leaf images cannot reliably distinguish between
            healthy leaves and ones containing powdery mildew better than random chance (**50%** accuracy).""")        

    st.info("""**Alternative Hypothesis:** A CNN model trained on cherry leaf images can reliably distinguish between
            healthy leaves and ones containing powdery mildew with at least a **97%** accuracy.""")
    
    st.success("""The null hypothesis has been rejected in favour of the alternative hypothesis.
               The final model was successfully trained using the cherry leaf dataset provided by the client.
               **Validation:** When evaluated on unseen test data, the model achieved an accuracy metric of **98.8%**, 
               meeting and exceeding the business requirement.      
            """)
