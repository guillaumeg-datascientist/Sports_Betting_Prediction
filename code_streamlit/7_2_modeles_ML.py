import streamlit as st
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble


#st.markdown("<style>.streamlit-expanderHeader {font-family:Rockwell; font-size: 26px; font-weight: bold}</style>",
#            unsafe_allow_html=True)

#with st.expander(label='second modèle'):
    
KNN = neighbors.KNeighborsClassifier(n_neighbors=3, metric='minkowski', weights='distance')
Decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=30)
Random_forest = ensemble.RandomForestClassifier(criterion='entropy', max_depth=30, n_estimators=80)

df = pd.read_csv("datasets_for_streamlit/atp_data.csv")


def Model(row, status, needed_data, classifier):
    clf = classifier

    list_of_prediction = []
    list_of_resultat = []
    list_of_row = []

    player = df.loc[row][status]
    df1 = df.iloc[0:row + 1]
    df_player = df1[(df1['Winner'] == player) | (df1['Loser'] == player)].sort_values(by=['Date'], ascending=True)

    df_player = df_player.drop(['Date', 'Location', 'Comment', 'ATP', 'PSW', 'PSL', 'B365W', 'B365L',
                                'Round', 'Wsets', 'Lsets', 'proba_elo'], axis=1)

    df_player['Winner'] = df_player['Winner'].replace(to_replace=player, value=1)
    df_player['Loser'] = df_player['Loser'].replace(to_replace=player, value=0)
    df_player['result'] = df_player.apply(lambda row: row['Winner'] if type(row['Winner']) == int else row['Loser'],
                                          axis=1)

    df_player = df_player.drop(['Winner', 'Loser'], axis=1)
    df_player = df_player.dropna(axis=0, how='any')

    target = df_player['result']
    data = df_player.drop('result', axis=1)

    data = data.join(pd.get_dummies(data[['Tournament', 'Series', 'Court', 'Surface']]))

    data = data.drop(['Tournament', 'Series', 'Court', 'Surface'], axis=1)

    if len(data) > needed_data:
        list_of_row.append(row)
        slice = int(len(data) * 0.75)

        for i in range(slice, len(data)):
            X_train = data.iloc[0:i]
            y_train = target.iloc[0:i]
            X_test = data.iloc[i:i + 1]
            y_test = np.array(target.iloc[i:i + 1])

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            list_of_prediction.append(y_pred[0])
            list_of_resultat.append(y_test)

    else:
        return f'Not enough data for the {status} player {player}, ' \
               f'Where the available informations are {len(data)} lines', 0, 0, 0, 0

    list = []

    for value1, value2 in zip(list_of_prediction, list_of_resultat):
        if value1 == value2:
            list.append(f'good prediction')
        elif value1 != value2:
            list.append(f'bad prediction')

    y_pred = clf.predict(data.tail(1))

    z = pd.DataFrame(list).value_counts(normalize=True)

    if y_pred[0] == 0:
        return f'avec un seuil de fidelité {round(z[0], 2) * 100} %', \
               f"l'algorithme prédit que le joueur {player} va perdre le match et la prédiction " \
               f"est basée sur {len(data)} lignes", 'lose', list_of_row
    else:
        return f'avec un seuil de fidelité {round(z[0], 2) * 100} %', \
               f"l'algorithme prédit que le joueur {player} va gagner le match et la prédiction " \
               f"est basée sur {len(data)} lignes", 'lose', list_of_row


st.markdown('<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">1. Rentabilité du Bookmaker</p>', unsafe_allow_html=True)

st.markdown("Dans cette étude nous allons utiliser quelque variables du dataset initial.")
st.markdown("Pour afficher quelque lignes du dataset")

if st.button('cliquer ici'):
    st.write(df.sample(5))

st.markdown("<h6 style='font-family:Helvetica ; color: purple;'>Peut-on faire confiance au bookmaker?</h6>",
            unsafe_allow_html=True)
st.markdown("supposons qu'un parrainage sera effectuer sur des matchs prévus gagnant par le bookmaker en mettant 1 euro sur chaque match")

#df = df.dropna(how='any', subset='B365W')
number_of_rows = st.slider(label='selectionner un nombre de matchs', min_value=10, max_value=35000, step=100)
df_select = df.sample(number_of_rows)
df_win = df_select[df_select['B365W'] < df_select['B365L']]
df_los = df_select[df_select['B365W'] > df_select['B365L']]
total_win = df_win['B365W'].sum() - (len(df_win) + len(df_los))

st.markdown(f"**Le gain total est : {round(total_win)} euros sur {number_of_rows} matchs**")
st.markdown("Nous pouvons conclure que le parrinage sur un ensemble des joueurs prévus gagnant par le bookmaker n'est pas rentable")

st.markdown('<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;"> 2. Modèle de prédiction </p>', unsafe_allow_html=True)

st.markdown("Dans cette partie nous avons créer un modèle de prédiction qui prédit le gagnant d'un match")

st.markdown("<h6 style='font-family:Helvetica ; color: purple;'>1. Description du modèle</h6>",
            unsafe_allow_html=True)
st.markdown("Les variables utilisées dans ce modèle sont:")
st.markdown('**Tournament**,  **Series**,  **Court**,  **Surface**,  **WRank**,  **LRank**,  **elo_winner**,  '
            '**elo_loser**')
st.markdown("la méthode consiste a prédire si le joueur va gagner un match ou non et cette prédiction est basés sur l'ensemble des matchs \
            joués par le joueur au paravant")

col1, col2, col3 = st.columns(3)

with col1:
    z = st.slider(label='choisir une ligne', min_value=10000, max_value=44700, step=1)

with col2:
    t = st.selectbox('quel joueur?', ['Winner', 'Loser'])

with col3:
    classifier = st.selectbox('Choisir un classifier', ['KNN', 'Decision_tree', 'Random_forest'])

if classifier == 'KNN':
    st.write(Model(row=z, status=t, needed_data=50, classifier=KNN)[1])
    st.write(Model(row=z, status=t, needed_data=50, classifier=KNN)[0])

elif classifier == 'Decision_tree':
    st.write(Model(row=z, status=t, needed_data=50, classifier=Decision_tree)[1])
    st.write(Model(row=z, status=t, needed_data=50, classifier=Decision_tree)[0])

elif classifier == 'Random_forest':
    st.write(Model(row=z, status=t, needed_data=50, classifier=Random_forest)[1])
    st.write(Model(row=z, status=t, needed_data=50, classifier=Random_forest)[0])

st.markdown("<h6 style='font-family:Helvetica ; color: purple;'>2. Application du modèle sur plusieurs observations</h6>", unsafe_allow_html=True)

st.markdown("Dans cette partie nous appliquons notre modèle sur des plusieurs observations")

dict1 = {"algo_score": ['     78 %', '     76 %', '     82 %'],
         "bookMaker_score": ['           76 %', '           76 %', '           76 %'],
         "algo_gain": ['154  euros', '147  euros', '204  euros'],
         "bookMaker_gain": ['      -5  euros', '      -5  euros', '      -5  euros']
         }

new_df = pd.DataFrame.from_dict(dict1)
new_df.index = ['Knn', 'Decision Tree', 'Random Forest']
st.markdown("<h8 style='font-family:Helvetica ; color: black;'>446 observations</h8>", unsafe_allow_html=True)
st.write(new_df.head())

dict2 = {"algo_score": ['     78 %', '     73 %', '     77 %'],
         "bookMaker_score": ['           74 %', '           74 %', '           74 %'],
         "algo_gain": ['1086  euros', '893  euros', '952  euros'],
         "bookMaker_gain": ['      -20  euros', '      -20  euros', '      -20  euros']
         }

new_df = pd.DataFrame.from_dict(dict2)
new_df.index = ['Knn', 'Decision Tree', 'Random Forest']
st.markdown("<h8 style='font-family:Helvetica ; color: black;'>3350 observations</h8>", unsafe_allow_html=True)
st.write(new_df.head())

st.markdown("<h6 style='font-family:Helvetica ; color: purple;'>Le KNN sur 7549 observations</h6>", unsafe_allow_html=True)

st.image('images_for_streamlit/streamlit.png')

