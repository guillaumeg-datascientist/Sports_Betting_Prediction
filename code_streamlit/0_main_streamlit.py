#import seaborn as sns
#import numpy as np
#import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt
#import altair as alt
#import plotly.express as px
#import datetime

from streamlit_option_menu import option_menu

#from sklearn import preprocessing
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.feature_selection import SelectKBest, mutual_info_regression
#from sklearn.metrics import confusion_matrix



with st.sidebar:
  selected = option_menu(menu_title = 'Sommaire', options = ['Titre et auteurs',
                                                             'Introduction', 
                                                             'Présentation des données',
                                                             'Nettoyage des données',
                                                             'Étude statistique',
                                                             'Pipeline',
                                                             'Modèle de prédiction',
                                                             'Conclusion'], default_index = 0, menu_icon='cast')
###on réccup tous les datasets

#df_ML = pd.read_csv('df.csv')
#df_ML.drop('Index', axis=1, inplace=True) #dataset utilisé pour les prédictions

#df = pd.read_csv('atp_data_4.csv', sep = ";") #dataset de base nettoyé

#df_players = pd.read_csv('atp_data_all_players_var.csv', sep=';').drop('Unnamed: 0', axis =1, inplace = True) #dataset avec tous les joueurs + variables
#df_players = df_players.set_index('Name')
#df_players['Date'] = pd.to_datetime(df_players['Date'], format = "%Y/%m/%d") #on rectifie le type de la colonne date

#####Partie 1

if selected == 'Titre et auteurs':

    exec(open('1_titre_et_auteurs.py').read())

elif selected == 'Introduction':

    exec(open("2_introduction.py").read())
    
elif selected == 'Présentation des données':

    exec(open("3_presentation_dataset.py").read())
    
elif selected == 'Nettoyage des données':

    exec(open("4_nettoyage.py").read())
    
elif selected == 'Étude statistique':

    exec(open("5_etude_statistique.py").read())

elif selected == 'Pipeline' :
    
    exec(open("6_1_pipeline.py").read())

elif selected == 'Modèle de prédiction' :
    
    exec(open("7_1_modeles_ML.py").read())
    
elif selected == 'Conclusion' :
    
    exec(open("8_conclusion_et_regard_critique.py").read())
    
    
    
    
    



    


        
        
        
    
    
    
    
    
    



