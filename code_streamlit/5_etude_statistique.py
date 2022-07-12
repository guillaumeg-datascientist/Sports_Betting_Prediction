#import seaborn as sns
#import numpy as np
import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt
#import altair as alt
import plotly.express as px
#import datetime

#from streamlit_option_menu import option_menu

#from sklearn import preprocessing
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.feature_selection import SelectKBest, mutual_info_regression
#from sklearn.metrics import confusion_matrix

df = pd.read_csv('datasets_for_streamlit/atp_data_4.csv', sep = ";") #dataset de base nettoyé

df_players = pd.read_csv('datasets_for_streamlit/atp_data_all_players_var.csv', sep=';')
df_players.drop('Unnamed: 0', axis =1, inplace = True)#dataset avec tous les joueurs + variables
df_players = df_players.set_index('Name')
df_players.drop(['Week_number','Year', 'Year_week'], axis=1, inplace=True)

df_players['Date'] = pd.to_datetime(df_players['Date'], format = "%Y/%m/%d") #on rectifie le type de la colonne date

st.title("Étude statistique")  
 
st.markdown("<style>.streamlit-expanderHeader {font-family:Rockwell; font-size: 26px; font-weight: bold}</style>",unsafe_allow_html=True)

#affichage des différentes statistiques des joueurs

with st.expander(label = 'Caractéristiques des joueurs', expanded = True):

#    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">1. Caractéristiques des joueurs</p>'
#    st.markdown(text, unsafe_allow_html=True)   

    ###fonctions create_crosstab
    @st.cache
    def create_crosstab(x, y, player_1, player_2, player_3, player_4, player_5, df_players) :
    
        stats_player_1 = df_players.loc[player_1,:].rename(columns = {x:'Player'})
        stats_player_2 = df_players.loc[player_2,:].rename(columns = {x:'Player'})
        stats_player_3 = df_players.loc[player_3,:].rename(columns = {x:'Player'})
        stats_player_4 = df_players.loc[player_4,:].rename(columns = {x:'Player'})
        stats_player_5 = df_players.loc[player_5,:].rename(columns = {x:'Player'})
        
        crosstab_1 = pd.crosstab(stats_player_1[y], stats_player_1['Player'], normalize = 0)
        crosstab_2 = pd.crosstab(stats_player_2[y], stats_player_2['Player'], normalize = 0)
        crosstab_3 = pd.crosstab(stats_player_3[y], stats_player_3['Player'], normalize = 0)
        crosstab_4 = pd.crosstab(stats_player_4[y], stats_player_4['Player'], normalize = 0)
        crosstab_5 = pd.crosstab(stats_player_5[y], stats_player_5['Player'], normalize = 0)
    
        crosstab_1 = crosstab_1.drop('D', axis = 1).rename(columns = {'V':player_1})
        crosstab_2 = crosstab_2.drop('D', axis = 1).rename(columns = {'V':player_2})
        crosstab_3 = crosstab_3.drop('D', axis = 1).rename(columns = {'V':player_3})
        crosstab_4 = crosstab_4.drop('D', axis = 1).rename(columns = {'V':player_4})
        crosstab_5 = crosstab_5.drop('D', axis = 1).rename(columns = {'V':player_5})
    
        crosstab_players = crosstab_1.merge(right=crosstab_2, on = y, how = 'outer')
        crosstab_players = crosstab_players.merge(right=crosstab_3, on = y, how = 'outer')
        crosstab_players = crosstab_players.merge(right=crosstab_4, on = y, how = 'outer')
        crosstab_players = crosstab_players.merge(right=crosstab_5, on = y, how = 'outer')
    
        crosstab_players = crosstab_players.reset_index().rename({y:' '}, axis = 1).set_index(' ')
    
        return crosstab_players
    
    ###fonctions create dataframe
    @st.cache
    def create_dataframe(bar, crosstab) :
        
        liste_var = list(crosstab.index)
        liste_df = []
    
        for var in liste_var :
            liste_var_copy = liste_var.copy()
            liste_var_copy.remove(var)
            df_bar = crosstab.transpose().reset_index().drop(liste_var_copy, axis = 1)
            df_bar = df_bar.rename(columns={var:'Pourcentage'})
            df_bar[bar] = [var]*5
            liste_df.append(df_bar)
    
        return pd.concat(liste_df, axis=0)    

    options = st.multiselect('Choix des joueurs :', df_players.index.value_counts().index, ['Federer R.', 'Cilic M.', 'Ferrer D.', 'Simon G.', 'Seppi A.'])
    
    player_1 = options[0]
    player_2 = options[1]
    player_3 = options[2]
    player_4 = options[3]
    player_5 = options[4]
    
    variable_choisi = st.selectbox(label = 'Choix de la variable', options = ['Surface', 'Court', 'Round', 'Format', 'Tournament'])
        
    ###fonction variable_barplot
        
    @st.cache
    def variable_barplot(variable_choisi, df_players) :
        
        if variable_choisi == 'Court' :
            
            crosstab_court = create_crosstab('Résultats', 'Court', player_1, player_2, player_3, player_4, player_5, df_players)
            vertical_concat = create_dataframe('Court', crosstab_court)
        
            fig = px.bar(vertical_concat, x='Court', y='Pourcentage', color = 'Player', barmode='group')
            fig.update_layout(title_font_size=24)
            fig.update_xaxes(showgrid=True)
            return fig
        
        elif variable_choisi == 'Surface' :
        
            crosstab_surface = create_crosstab('Résultats', 'Surface', player_1, player_2, player_3, player_4, player_5, df_players)
            vertical_concat = create_dataframe('Surface', crosstab_surface)
        
            fig = px.bar(vertical_concat, x='Surface', y='Pourcentage', color = 'Player', barmode='group')
            fig.update_layout(title_font_size=24)
            fig.update_xaxes(showgrid=True)
            return fig
        
        elif variable_choisi == 'Round' :
        
            df_players_round_GS = df_players[df_players['Series']=='GS']
            crosstab_round_GS = create_crosstab('Résultats', 'Round', player_1, player_2, player_3, player_4, player_5, df_players_round_GS)
            crosstab_round_GS = crosstab_round_GS.transpose()[['1R', '2R', '3R', '8th', 'Q', 'SF', 'F']].transpose()
            vertical_concat = create_dataframe('Round', crosstab_round_GS)
        
            fig = px.bar(vertical_concat, x='Round', y='Pourcentage', color = 'Player', barmode='group')
            fig.update_layout(title_font_size=24)
            fig.update_xaxes(showgrid=True)
            return fig
        
        elif variable_choisi == 'Format' :
            
            crosstab_format = create_crosstab('Résultats', 'Format', player_1, player_2, player_3, player_4, player_5, df_players)
            vertical_concat = create_dataframe('Format', crosstab_format)
            
            fig = px.bar(vertical_concat, x='Format', y='Pourcentage', color = 'Player', barmode='group')
            fig.update_layout(title_font_size=24)
            fig.update_xaxes(showgrid=True)
            return fig
        
        elif variable_choisi == 'Tournament' :
        
            df_players_M1000 = df_players[df_players['Series']=='M1000']
        
            crosstab_m1000 = create_crosstab('Résultats', 'Tournament', player_1, player_2, player_3, player_4, player_5, df_players_M1000)
            crosstab_m1000.drop(['Stuttgart Master', 'Paris Master IC', 'Hamburg Master', 'Madrid Master IH', 'New York Master'], axis = 0, inplace=True)
            vertical_concat = create_dataframe('Tournament', crosstab_m1000)
        
            fig = px.bar(vertical_concat, x='Tournament', y='Pourcentage', color = 'Player', barmode='group')
            fig.update_layout(title_font_size=24)
            fig.update_xaxes(showgrid=True)
            return fig
        
    st.plotly_chart(variable_barplot(variable_choisi, df_players))
       
    st.markdown("L'étude statistique de ces variables met en lumière le fait que chaque joueur est unique avec des caractéristiques différentes : ")
    
    st.markdown("* Des joueurs que l'on peut catégoriser de par leurs résultats de manière générale : \
                joueur de seconde zone, bon joueur, très bon joueur ou top players.")
                
    st.markdown("* Des joueurs avec des styles de jeu différents, avec des joueurs davantage spécialisés en indoor ou outdoor, sur terre battue ou sur gazon.")
    st.markdown("* Des joueurs avec des taux de réussites différents d'un tour à l'autre selon les catégories de tournoi.")
    st.markdown("* Des joueurs possédant des tournois fétiches dans lesquels les joueurs ont davantage performés par le passé.")
    st.markdown(" ")
                
    
        ###fonction h2h_barplot
       
with st.expander(label = 'Face à face entre les joueurs'):
       
        
    @st.cache
    def h2h_barplot(player_1, player_2, df) :
        
        df_h2h = df[ ((df['Winner']==player_1) & (df['Loser']==player_2)) |
                     ((df['Winner']==player_2) & (df['Loser']==player_1)) ]
        
        df_h2h_win = df_h2h.drop('Loser', axis = 1)
        df_h2h_lose = df_h2h.drop('Winner', axis =1)
        
        df_h2h_win = df_h2h_win.assign(Victoire=1, Défaite=0)
        df_h2h_lose = df_h2h_lose.assign(Victoire=0, Défaite=1)
        
        df_h2h_win = df_h2h_win.rename({'Winner':'Player', }, axis = 1)
        df_h2h_lose = df_h2h_lose.rename({'Loser':'Player'}, axis = 1)
        
        df_players_h2h = pd.concat([df_h2h_win, df_h2h_lose])
        df_players_h2h = df_players_h2h.groupby(['Surface', 'Player']).sum().reset_index()
        df_players_h2h.drop('Défaite', axis=1, inplace=True)
        
        fig = px.bar(df_players_h2h, x="Player", y="Victoire", color="Surface", color_discrete_map={'Clay':'darkorange',
                                                                                                    'Grass':'forestgreen',
                                                                                                    'Hard':'cornflowerblue','Carpet':'plum'})
        
        return fig



    #text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">2. Face à face entre les joueurs</p>'
    #st.markdown(text, unsafe_allow_html=True)   

    options = st.multiselect('Choix des players :', df_players.index.value_counts().index, ['Simon G.', 'Cilic M.' ])
    
    player_1 = options[0]
    player_2 = options[1]
    
    st.plotly_chart(h2h_barplot(player_1, player_2, df))
    
    st.markdown("* Les face-à-face entre 2 joueurs montrent que l'on ne peut pas comparer seulement les statistiques entre 2 joueurs. \
                Même si les statistiques jouent en la faveur d'un joueur, il se peut que celui-ci ait un bilan défavorable face à \
                son adversaire. Ces cas peuvent être causés par un manque de solutions tactiques ou des styles de jeu incompatibles.")
                
    st.markdown("* Les face-à-face sur différentes surfaces sont également intéressants car un bilan positif d'un joueur sur une surface \
                peut être négatif sur une autre surface.")
                


    
    
    
    
