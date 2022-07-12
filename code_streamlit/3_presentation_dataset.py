import streamlit as st
import pandas as pd
from PIL import Image

st.title('Présentation des données')

st.markdown("<style>.streamlit-expanderHeader {font-family:Rockwell; font-size: 26px; font-weight: bold}</style>",unsafe_allow_html=True)

#Dans la fonction expander, expanded = True permet de spécifier que l'on souhaite avoir le l'expander déjà déroulé

with st.expander(label = '1. Inspiration', expanded = True): 
    

    st.markdown("*https://www.kaggle.com/datasets/edouardthomas/atp-matches-dataset*")
    image = Image.open('images_for_streamlit/kaggle.png')
    st.image(image, caption='Dataset proposé dans la fiche projet Datascientest')
    
        
    st.markdown(" \n")
    st.markdown(" \n")
    
    
with st.expander(label = '2. Source initiale'):

    st.markdown("*http://tennis-data.co.uk/alldata.php*")
    image = Image.open('images_for_streamlit/tennisdata.png')
    st.image(image, caption='Datasets annuels que nous avons recompilés en un dataset unique')
    

        
    st.markdown(" \n")
    st.markdown(" \n")

    
with st.expander(label = '3. Dataset initial'):


    st.markdown("*'atp_data_1.csv' (+ de 58K matches)'*")
    atp_data_1 = pd.read_csv("datasets_for_streamlit/atp_data_1_sample.csv", sep = ";")
    st.dataframe(atp_data_1)
    
    
#Ici on ne veut pas que l'expander soit déjà déroulé

    
    
    # streamlit run C:\Users\guiga\presentation_dataset.py
