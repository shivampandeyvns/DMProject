import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import keras
from streamlit_player import st_player 

def names(number):
    if number==0:
        return 'a Tumor'
    else:
        return 'not a tumor'

def classifier():
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        model=keras.models.load_model('CNN_Brain_tumor2.h5')
        
        x = np.array(image.resize((128,128)))
        x = x.reshape(1,128,128,3)
        res = model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        st.subheader(str(res[0][classification]*100) + '% Confidence This Is ' + names(classification))


st.title('Brain Tumor Classifier')

image=Image.open('download.jfif')
st.image(image)

menu=['Home','Classifier','About']
choice=st.sidebar.selectbox('Menu',menu)

check=False

if(choice=='Home'):

    st.subheader('What is a Brain Tumor\n')
    st.markdown("""\n
        A cancerous or non-cancerous mass or growth of abnormal cells in the brain.
        Tumours can start in the brain, or cancer elsewhere in the body can spread to the b rain.
        Symptoms include new or increasingly strong headaches, blurred vision, loss of balance, confusion and seizures. In some cases, there may be no symptoms.
        Treatments include surgery, radiation and chemotherapy.
    """)

    st.subheader('Understanding Brain Tumor\n\n')
    st_player('https://www.youtube.com/watch?v=pBSncknENRc')

    st.subheader('Symptoms Of Brain Tumor\n')
    st.markdown("""
        \n\n
         1.Headache\n
         2.Muscle Weakness\n
         3.Nausea\n
         4.Vomitting\n
         5.Dizziness\n
         6.Inability To speak\n
         7.Mental Confusion\n

    """)

    st.subheader('TreatMent')
    st.markdown("""
        \n\n\
        1.**Chemotherapy**->Unwanted reactions to drugs given for the purpose of killing cancer cells.
        2.**Craniotomy**->Brain surgery in which a piece of bone is removed from the skull.
        3.**Tomotherapy**->Cancer treatment that aims high-dose radiation at tumours from many directions. Reduces damage to nearby tissue.    
        4.**Radiation therapy**->Treatment that uses x-rays and other high-energy rays to kill abnormal cells.
    """)

    st.write("**For More Information** [Click Here](https://www.cancer.net/cancer-types/brain-tumor/symptoms-and-signs)")
elif(choice=='Classifier'):
    st.subheader('How To Use the App')
    st.write('Just Upload your **MRI scan** and the Classifier will tell you whether you have Tumor or not with a Confidence Percentage')
    classifier()


elif(choice=='About'):
    st.markdown("""
        The App is Build Purely in Python Programming Language.\n
        This App Uses Convolutional Neural Network for the Image Classification.\n
        Python's Deep Learning Library **Keras** was used in this App, which uses **Tensorflow** in the backend.\n
        The Dataset was Obtained From Kaggle->[Click Here](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)\n
    """)
    expander=st.expander('Group Members For the Project')
    expander.markdown("""
        **Alinjar Das**\n
        **Abhijeet Padhi**\n
        **Sidharth Kumar Choudhary**\n
        **Amit Prakash**\n
        **Mayank Sinha**\n
        **Shivam Pandey**\n
    """)
