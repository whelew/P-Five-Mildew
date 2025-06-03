import streamlit as st

def page_summary():
    st.title('Proejct Summary')

    st.subheader('General Information')
    st.write(""" The company Farmy & Foods is facing an issue with powdery mildew affecting their 
             cherry plantations. The current process to identify whether powdery mildew is present within
             a cherry tree is a manual one requiring at least 30 minutes of an employees time. If there is 
             powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying 
             this compound is 1 minute. The company has thousands of trees across multiple farms, this process 
             is time consuming and not scalable.
        """)

    st.subheader('Client Goal')
    st.write(""" The client has requested a machine learning solution that can instantly detect
             powdery mildew in cherry leaves using image data.
        """)
    
    st.subheader('Business Requirements')
    st.write('1. The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.')
    st.write('2. The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.')