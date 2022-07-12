import pandas as pd
import streamlit as st

df_players = pd.read_csv('datasets_for_streamlit/atp_data_all_players_var.csv', sep=';')
df_players.drop('Unnamed: 0', axis =1, inplace = True)#dataset avec tous les joueurs + variables
df_players = df_players.set_index('Name')
df_players.drop(['Week_number','Year', 'Year_week'], axis=1, inplace=True)

df_players['Date'] = pd.to_datetime(df_players['Date'], format = "%Y/%m/%d") #on rectifie le type de la colonne date

text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">1. Initialisation des variables</p>'
st.markdown(text, unsafe_allow_html=True)

st.markdown("Création d'un dataset annexe répertoriant tous les matchs joués par les joueurs. Chaque ligne correspond au match joué par \
            le joueur avec toutes les informations associées au match ainsi que l'issue du match renseignée dans la colonne **Résultats** : \
            *V* pour une victoire et *D* pour une défaite.")

st.dataframe(df_players[['Date', 'Location', 'Tournament', 'Series', 'Court', 'Surface', 'Round', 'Format', 'Résultats']].sample(5))

#st.dataframe(df_players)

#affichage d'un dataframe comprenant tous les matchs d'un joueur entre 2 dates

st.markdown("L'intérêt de ce dataset est que nous pouvons avoir accès à chacun des matchs joués par le joueur sur une période donnée. \
            Impliquant ainsi que pour chacun des matchs, nous pouvons faire remonter toutes les informations que nous souhaitons sur ses matchs \
            passés et ce sur la période souhaitée.")

player = st.text_input('Nom du joueur', 'Federer R.')

col11, col12 = st.columns(2)

date_debut = col11.text_input('Date début :', '2002-02-01')
date_fin = col12.text_input('Date fin :', '2002-04-01')

df_players_temp = df_players.loc[player, :]
df_players_temp = df_players_temp[ (df_players_temp['Date'] >= pd.to_datetime(date_debut)) & (df_players_temp['Date'] <= pd.to_datetime(date_fin))]

st.dataframe(df_players_temp[['Date', 'Location', 'Tournament', 'Series', 'Court', 'Surface', 'Round', 'Format', 'Résultats']].head(30))

col21, col22 = st.columns(2)

variable_input = col21.text_input('Variable :', 'Surface')
modalite_input = col22.text_input('Modalité :', 'Clay')

@st.cache
def calcul_variable(variable, modalite) :
    return len(df_players_temp[ (df_players_temp[variable] == modalite) & (df_players_temp['Résultats'] == 'V')])


st.markdown('Le nombre de matchs remportés par **%s** entre le **%s** et le **%s** pour la variable **%s** et la modalité **%s** est : %d' %(player, date_debut, date_fin, 
                                                                                                         variable_input, modalite_input,
                                                                                                         calcul_variable(variable_input, modalite_input)))
         
   
text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">2. Dataset annexe avec variables implémentées</p>'
st.markdown(text, unsafe_allow_html=True)

st.dataframe(df_players_temp)
