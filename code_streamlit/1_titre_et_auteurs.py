import streamlit as st
from PIL import Image


st.title('How to get the Bookies grum.py')
st.markdown('<center> \
                Auteurs : \
                    <b> \
                        Guillaume Gréau, Guillaume Oillo, Georges Sleimen \
                    </b> \
            </center>', unsafe_allow_html=True)

image = Image.open('images_for_streamlit/cover.png')
st.image(image)