import pandas as pd
import numpy as np
import streamlit as st

df = pd.read_csv('datasets_for_streamlit/atp_data_2.csv', sep = ";")
df.drop(['Unnamed: 0'], axis = 1, inplace=True)

st.title("Nettoyage des données")  
 
st.markdown("Avant tout changement ou toute correction dans le dataset, la présence de valeurs non renseignées et de lignes dupliquées \
             a été vérifée. Pas de lignes dupliquées et les valeurs non renseignées ne concernent que les variables que nous n'avons pas utilisé \
             par la suite.")
#st.markdown("<style>.streamlit-expanderHeader {font-family:Rockwell; font-size: 26px; font-weight: bold}</style>",unsafe_allow_html=True)

###1###

text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">1. Remplacement des anciens noms des catégories</p>'
st.markdown(text, unsafe_allow_html=True)

st.markdown("Modalités de la variable **Series** :")

st.table(df['Series'].value_counts())

st.markdown("Les noms des catégories des tournois ont changé au cours du temps comme par exemple la catégorie **International** qui \
            correspond maintenant à un ATP 250.")

df = df.replace(to_replace=['International Gold', 'International', 'Masters'], value = ['ATP500', 'ATP250', 'Masters 1000'])
    
with st.expander(label = 'Après remplacement'):
        
        st.table(df['Series'].value_counts())

###2###

text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">'+"2. Utilisation d'abréviations </p>"
st.markdown(text, unsafe_allow_html=True)

st.markdown("Modalités des variables **Series** et **Round** :")

col11, col12 = st.columns(2)

col11.table(df['Round'].value_counts())
col12.table(df['Series'].value_counts())

st.markdown("Par soucis de commodité, les noms des modalités des variables **Series** et **Round** ont été raccourcis.")

df = df.replace(to_replace = ['1st Round', '2nd Round', '3rd Round', '4th Round', 
                              'Quarterfinals', 'Semifinals', 'The Final', 'Round Robin',
                              'Masters 1000', 'Grand Slam', 'Masters Cup'],
                value = ['1R', '2R', '3R', '8th', 'Q', 'SF', 'F', 'RR', 'M1000', 'GS', 'ATP Final'])

with st.expander(label = 'Après renommage'): 
    
    col21, col22 = st.columns(2)
    
    col21.table(df['Round'].value_counts())
    col22.table(df['Series'].value_counts())

###3###

text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">3. Suppression de typographie</p>'
st.markdown(text, unsafe_allow_html=True)

st.markdown("Modalités de la variable **Location** lorsque **Tournament**==*Dubai Championship* :")

st.table(df[ (df['Tournament']=='Dubai Championships') | (df['Tournament']=='Dubai Open') ]['Location'].value_counts())

st.markdown("Certaines localisations de tournois comme le tournoi de *Dubai Championship* apparaissent dans 2 localisations *Dubai* différentes. \
            L'une d'entre elles a été inscrite avec un espace à la fin.")

df = df.replace(to_replace = ['Dubai ', 'Estoril ', 'Vienna ', "'s-Hertogenbosch", "Winston-Salem", "Nur-Sultan"], 
                     value = ['Dubai', 'Estoril', 'Vienna', "Hertogenbosch", "Winston Salem", "Nur Sultan"])
    
with st.expander(label = 'Après correction'):
    
    st.table(df[ (df['Tournament']=='Dubai Championships') | (df['Tournament']=='Dubai Open') ]['Location'].value_counts())

###4###

text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">4. Regroupement de tournois sous un même nom</p>'
st.markdown(text, unsafe_allow_html=True)

st.markdown("Modalités de la variable **Tournament** lorsque **Location**==*Indian Wells* :")

st.table(df[ df['Location'] == 'Indian Wells']['Tournament'].value_counts())

st.markdown("La majorité des tournois ont subit des changements de noms au cours des années comme avec le tournoi \
            d'*Indian Wells* qui s'est joué sous 3 appellations différentes. L'astuce a été d'utiliser la localisation pour les regrouper \
            sous un même nom.")

df.loc[df['Tournament']=='Great Ocean Road Open', ['Location']] = 'Melbourne Great Ocean Road'
df.loc[df['Tournament']=='Murray River Open', ['Location']] = 'Melbourne Murray River'

df['Tournament'] = df['Location'] + '-' + df['Series']

def rename_tournament(tournament) :
    
    tournament = tournament.split('-')
    if (tournament[1] == 'ATP250') or (tournament[1] == 'ATP500') :
        return tournament[0] + ' Championship'
    elif (tournament[1] == 'M1000') :
        return tournament[0] + ' Master'
    elif (tournament[1] == 'GS') :  
        if (tournament[0] == 'Paris') :
            return 'Roland Garros'
        elif (tournament[0] == 'London') :
            return 'Wimbledon'
        elif (tournament[0] == 'New York') :
            return 'US Open'
        elif (tournament[0] == 'Melbourne') :
            return 'Australian Open'
    elif (tournament[1] == 'ATP Final') :
        return tournament[0] + ' ATP Final'
    
df['Tournament'] = df['Tournament'].apply(rename_tournament)
    
with st.expander(label = 'Après renommage'):

    st.table(df[ df['Location'] == 'Indian Wells']['Tournament'].value_counts())

###5###

text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">5. Distinction des tournois avec changement de surface</p>'
st.markdown(text, unsafe_allow_html=True)

st.markdown("Modalités de la variable **Surface** lorsque **Tournament**==*Madrid Master*")

st.table(df[df['Tournament'] == 'Madrid Master']['Surface'].value_counts())

st.markdown("Certains tournois ont conservé la même appellation mais ont subi un changement de court ou de surface. Il est nécessaire de les \
            différencier.")

df['Tournament'] = df['Tournament'] + '-' + df['Court'] + '-' + df['Surface']

tournament = list(df['Tournament'].value_counts().index.sort_values(ascending=True))

for i, t in enumerate(tournament) :
    tournament[i] = t.split('-')

name_city = np.array(tournament)[:,0]

import collections
doublons = [item for item, count in collections.Counter(name_city).items() if count > 1]

doublons_tournament = []
new_doublons_tournament = []

for i, t in enumerate(tournament) : 
    if t[0] in doublons :
        doublons_tournament.append(t[0]+'-'+t[1]+'-'+t[2])
        new_name = t[0]+' '+ t[1][0] + t[2][0]+'-'+t[1]+'-'+t[2]
        new_doublons_tournament.append(new_name)
        
df = df.replace(to_replace = doublons_tournament, value = new_doublons_tournament)

df['Tournament'] = df['Tournament'].apply(lambda tournament : str(tournament).split('-')[0])

with st.expander(label = 'Après renommage'):

    st.table(df[df['Location'] == 'Madrid']['Tournament'].value_counts())

###6###

text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">6. Remplacement des noms des tours selon les catégories</p>'
st.markdown(text, unsafe_allow_html=True)

df.loc[df['Location']=='Melbourne Great Ocean Road', ['Location']] = 'Melbourne'
df.loc[df['Location']=='Melbourne Murray River', ['Location']] = 'Melbourne'

st.markdown("Table fabriquée qui répertorie tous les formats de tournois : ")

df_round = df.groupby(['Year', 'Tournament', 'Series', 'Round']).count().loc[:,['Court']]
df_round = df_round.reset_index()
df_round['id Tournament'] = df_round['Year'].astype(str) +'-'+df_round['Tournament']+'-'+df_round['Series']
df_round = df_round.drop(['Year', 'Tournament', 'Series'], axis = 1)

all_tournaments = list(df_round['id Tournament'].value_counts().index.sort_values(ascending=True))
df_round = df_round.set_index('id Tournament')

def concat_round_nb_round(data, id_tournament, type_round, nb_matchs_round) :

    df_int = data.loc[[id_tournament],:].sort_values(by = nb_matchs_round, ascending = False)

    Round = ''
    Court = ''
    Nb_tours = 0

    for element1, element2 in zip(df_int[type_round], df_int[nb_matchs_round]) :
        Round += element1+'/'
        Court += str(element2)+'/'
        Nb_tours += element2

    return [Round[:-1], Court[:-1], Nb_tours]

data = []

for tournament in all_tournaments :
    data.append(concat_round_nb_round(df_round, tournament, 'Round', 'Court'))
    
infos_tournament = pd.DataFrame(data = data,
                                index = all_tournaments,
                                columns = ['Round', 'Nb_matchs_round', 'Nb_matchs_tournament'])

st.table(infos_tournament.loc[['2000-Adelaide Championship-ATP250',
                               '2000-Australian Open-GS']])

st.markdown("Selon les catégories, l'enchaînement des tours au fil de la compétition n'est pas le même car certains tournois ne comportent \
            pas le même nombre de tours. Le tournoi d'*Adelaide* ne comporte que 5 tours contre 7 pour l'*Australian Open* et le *3R* et le *8th* \
            ne sont pas présents durant le tournoi d'*Adelaide*. Pour la suite, il est nécessaire que cet enchaînement soit cohérent \
            d'un tournoi à l'autre. Si le tournoi ne comporte que 5 tours, alors celui-ci doit démarrer par le 3R.")
    
tournament_type_2 = list(infos_tournament[(infos_tournament['Round'] == '1R/2R/3R/Q/SF/F') |
                                          (infos_tournament['Round'] == '2R/1R/3R/Q/SF/F')].index)

tournament_type_3 = list(infos_tournament[infos_tournament['Round'] == '1R/2R/Q/SF/F'].index)

df['Year-Tournament'] = df['Year'].astype(str) + '-' + df['Tournament'] + '-' + df['Series']

df = df.set_index('Year-Tournament')

df.loc[tournament_type_2, ['Round']] = df.loc[tournament_type_2, ['Round']].replace(to_replace = ['1R', '2R', '3R'],
                                                                                    value = ['2R', '3R', '8th'])

df.loc[tournament_type_3, ['Round']] = df.loc[tournament_type_3, ['Round']].replace(to_replace = ['1R', '2R'],
                                                                                    value = ['3R', '8th'])
df = df.reset_index().drop('Year-Tournament', axis = 1)

with st.expander(label = 'Après renommage'):
    
    df_round2 = df.groupby(['Year', 'Tournament', 'Series', 'Round']).count().loc[:,['Court']]
    df_round2 = df_round2.reset_index()
    df_round2['id Tournament'] = df_round2['Year'].astype(str) +'-'+df_round2['Tournament']+'-'+df_round2['Series']
    df_round2 = df_round2.drop(['Year', 'Tournament', 'Series'], axis = 1)
    
    all_tournaments2 = list(df_round2['id Tournament'].value_counts().index.sort_values(ascending=True))
    df_round2 = df_round2.set_index('id Tournament')

    data2 = []

    for tournament in all_tournaments :
        data2.append(concat_round_nb_round(df_round2, tournament, 'Round', 'Court'))

    infos_tournament2 = pd.DataFrame(data = data2,
                                    index = all_tournaments2,
                                    columns = ['Round', 'Nb_matchs_round', 'Nb_matchs_tournament'])
        
    st.table(infos_tournament2.loc[['2000-Adelaide Championship-ATP250',
                                   '2000-Australian Open-GS']])

        

#    all_tournaments = list(df_round['id Tournament'].value_counts().index.sort_values(ascending=True))
#    df_round = df_round.set_index('id Tournament')

#    for tournament in all_tournaments :
#        data.append(concat_round_nb_round(df_round, tournament, 'Round', 'Court'))
        
#    infos_tournament = pd.DataFrame(data = data,
#                                    index = all_tournaments,
#                                    columns = ['Round', 'Nb_matchs_round', 'Nb_matchs_tournament'])
    
#    st.table(infos_tournament.loc[['2000-Adelaide Championship-ATP250',
#                                   '2000-Australian Open-GS']])

        
    
    
    
























