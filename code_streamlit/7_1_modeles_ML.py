#import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt
#import altair as alt
import plotly.express as px
#import datetime

#from streamlit_option_menu import option_menu

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import confusion_matrix

df_ML = pd.read_csv('datasets_for_streamlit/df.csv')
df_ML.drop('Index', axis=1, inplace=True) #dataset utilisé pour les prédictions

df = pd.read_csv('datasets_for_streamlit/atp_data_4.csv', sep = ";") #dataset de base nettoyé

#df_players = pd.read_csv('atp_data_all_players_var.csv', sep=';').drop('Unnamed: 0', axis=1) #dataset avec tous les joueurs + variables
#df_players = df_players.set_index('Name')
#df_players['Date'] = pd.to_datetime(df_players['Date'], format = "%Y/%m/%d") #on rectifie le type de la colonne date

st.title('Modèles de prédiction')

st.markdown("<style>.streamlit-expanderHeader {font-family:Rockwell; font-size: 26px; font-weight: bold}</style>",unsafe_allow_html=True)

with st.expander(label = 'Étude préliminaire des variables', expanded = True): 
    
    image = st.selectbox("Choix de l'étude :", options = ['Corrélations entre les variables et la target /année', 
                                                          'Taux de prédictions réalisables /variable et /année'])
    
    if image == 'Corrélations entre les variables et la target /année' :
        
        st.image('images_for_streamlit/corr_taux_reussite.png')
        
        st.markdown("* On retrouve certains groupes de variables qui semblent très corrêlées entre elles : les variables relatives au classement, \
                    aux face-à-face, au calcul victoires-défaites et celles du calcul ratio victoires/nombre de matchs joués.")
        st.markdown("* Les variables relatives au classement ATP obtiennent les meilleurs résultats avec un taux de 0.65 en moyenne et jusqu'à \
                    0.68 pour certaines années.")
        st.markdown("* De très mauvais résultats sont obtenus pour les variables **V-D_rang_delta**, **since_last_match_played_data**.")
        st.markdown("* Les variables qui concernent les résultats observés sur les 3 dernières semaines ou sur les tournois passés \
                    semblent avoir un impact sur l'issue du match mais celui-ci est relativement plus faible que ceux dont le taux est supérieur à 0.6.")
        st.markdown(" ")
        
    elif image ==  'Taux de prédictions réalisables /variable et /année' :
        
        st.image('images_for_streamlit/taux_var_indeter.png')
        
        st.markdown("* Le taux de prédictions réalisables est très élevé pour toutes les variables relatives au classement, calcul victoires-défaites \
                    et celles du calcul ratio victoires/nombre de matchs joués.")
        st.markdown("* Le taux est faible pour les variables des face-à-face car il y a un grand nombre de matchs pour lesquels les 2 joueurs \
                    ne se sont jamais rencontrés auparavant.")
        st.markdown("* Les taux sont particulièrement plus faibles pour les premières années et surtout l'année 2000 car toutes les variables sont \
                    initialisées cette année-là et elles ne prennent pas en compte ce qui s'est déroulé avant 2000.")
        st.markdown("* La période qui semble la plus intéressante en terme de richesse statistiques semble être entre l'année 2005 et l'année 2021.")
        st.markdown(" ")
    
    
with st.expander(label = 'Premier modèle'):
    
    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">1. Choix du modèle</p>'
    st.markdown(text, unsafe_allow_html=True)  

    #découpage du dataset entre 2005 et 2020
    
    st.markdown("* Une première sélection du dataset a été effectué entre 2005 et 2020 avant la période du covid afin d'avoir la statistique la \
                plus représentative possible.")
    st.markdown("* Le choix du train et test set a été de conserver 80% de données d'entraînement et 20% de données de test. Le découpage a été \
                effectué de manière temporelle.")
    st.markdown(" ")
    
    start = 14518
    end = 55459
    
    #séparation du dataset en train et test
    
    df_ML = df_ML.iloc[start:end,:]
    
    size_dataset = len(df_ML)
    proportion_train = 80
    
    size_train = size_dataset*proportion_train//100
    
    data = df_ML.drop('target', axis = 1)
    target = df_ML['target']
    
    X_train = data.loc[:start+size_train-1,:]
    y_train = target.loc[:start+size_train-1]
     
    X_test = data.loc[start+size_train:,:]
    y_test = target.loc[start+size_train:]
    
    st.markdown("Un exemple du X_train avant standardisation :")
    
    st.dataframe(X_train.sample(2))
    
    #rescale train et test
    
    scaler = preprocessing.MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    
    #affichage du X_train rescale
    
    st.markdown("Un exemple du X_train après standardisation :")
    
    st.dataframe(X_train.sample(2))
    
    #Modèles de prédiction
    
    model_names = ['RandomForestClassifier', 'KNeighborsClassifier',
                   'SVM', 'DecisionTreeClassifier'] 
    
    model = st.radio('Choix du modèle :', model_names)
    
    if model == 'RandomForestClassifier' :
        
        max_features = st.selectbox('max_features :', options = ['log2', 'sqrt', None])
    
        min_samples_split = st.select_slider('min_samples_split :', options=range(2, 40, 1), value=38)
        
        clf = RandomForestClassifier(n_jobs=-1, max_features = max_features, 
                                 min_samples_split = min_samples_split, random_state = 1)
    
    elif model == 'KNeighborsClassifier' :
        
        n_neighbors = st.select_slider('n_neighbors :', options = range(2,20), value=10)
        metric = st.selectbox('metric :', options = ['manhattan', 'minkowski'])
    
        clf = KNeighborsClassifier(metric = metric, n_neighbors = n_neighbors) 
    
    elif model == 'SVM' :
    
        C = st.selectbox('C :', options = [0.1, 1, 10])
        kernel = st.selectbox('kernel :', options = ['rbf', 'linear', 'poly'])
        gamma = st.selectbox('gamma :', options = [0.001, 0.1, 0.5])
    
        clf = svm.SVC(C = C, gamma = gamma, kernel = kernel)
    
    else :
        
        criterion = st.selectbox('entropy :', options = ['entropy', 'gini'])
        max_depth = st.select_slider('max_depth :', options = range(1, 20), value = 10)
    
        clf = DecisionTreeClassifier(criterion = 'gini', max_depth = 5)
    
    
    k = st.select_slider('nombre de variables :', options=range(1, len(X_train.columns)+1, 1), value=23)
    
    ###fonction score_model
    
    @st.cache
    def score_model(X_train, X_test, y_train, y_test, model, nb_var) :
        
        sel = SelectKBest(score_func =  mutual_info_regression, k=nb_var) 
        sel.fit(X_train, y_train)
        mask = sel.get_support()
        
        X_train_sel = X_train[X_train.columns[mask]]
        X_test_sel = X_test[X_test.columns[mask]]
    
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)
        
        score = model.score(X_test_sel, y_test)
        
        return score, y_pred, mask
    
    #Calcul du score du modèle de prédiction
    
    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">2. Performances du modèles</p>'
    st.markdown(text, unsafe_allow_html=True)  
    
    score, y_pred, mask = score_model(X_train, X_test, y_train, y_test, clf, k)
    
    st.write('Le score est de : %s '%round(score,5))
            
    if st.checkbox('Afficher la matrice de confusion') :
        dataframe = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(dataframe, columns=np.unique(y_test), index = np.unique(y_test))/len(y_test)
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        
        fig = px.imshow(df_cm, text_auto=True)
        fig.update_layout(autosize=False, width=500, height=500)
        
        st.plotly_chart(fig)
        
    if st.checkbox('Afficher les variables sélectionnées') :
        data.columns[mask]
          
    st.markdown("* Les meilleurs résultats obtenus sont avec le RandomForestClassifier pour paramètres : max_features = *log2* et \
                min_samples_split = *38*. Ces résultats sont obtenus avec conservation de toutes les variables. En faisant varier le nombre \
                de variables, il est possible de maximiser légérement le score obtenu.")
    st.markdown("* Les scores obtenus oscillent entre 0.647 et 0.65 avec ce modèle donc un pourcentage de réussite d'environ 65% contre \
                67% pour le bookmaker Bet365 qui est celui possédant les meilleurs résultats.") 
    st.markdown("")
                
        
    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">Étude du taux de prédictions réussies suivant les modalités\
        de différentes variables</p>'
    st.markdown(text, unsafe_allow_html=True)  
    
    #réccup toutes les lignes avec de mauvaises prédictions
    
    df_all = df.loc[start:end,:]
    df_test = df_all.loc[start+size_train:,:]
    y_test_pred = pd.Series(y_pred+np.array(y_test))
    
    df_test = df_test.reset_index().drop(['Unnamed: 0'], axis =1)
    
    df_bad_pred = pd.concat([df_test, pd.DataFrame(y_test_pred)], axis=1)
    
    df_bad_pred = df_bad_pred[df_bad_pred[0]==1]
    
    #études des bonnes prédictions sur différentes variables
    
    bar = ['Catégories', 'Surfaces', 'Tours Grand Chelem', 'Tours Master 1000', 'Meilleurs tournois', 'Pires tournois']
    
    variable = st.selectbox('Variable :', options = bar)
     
    if variable == 'Surfaces' :
            
        df_surface = pd.DataFrame(1-(df_bad_pred['Surface'].value_counts()/df_test['Surface'].value_counts()))
        df_surface = df_surface.reset_index().rename(columns = {'Surface':'Pourcentage', 'index' : 'Surface'})
    
        fig = px.bar(df_surface, x='Surface', y='Pourcentage')
        st.plotly_chart(fig)
            
    elif variable == 'Catégories' :
            
        df_series = pd.DataFrame(1-(df_bad_pred['Series'].value_counts()/df_test['Series'].value_counts()))
        df_series = df_series.reset_index().rename(columns = {'Series':'Pourcentage', 'index' : 'Series'})
            
        fig = px.bar(df_series, x='Series', y='Pourcentage')
        st.plotly_chart(fig)
     
    elif variable == 'Tours Grand Chelem' :
            
        df_bad_pred_GS = df_bad_pred[ df_bad_pred['Series'] == 'GS']
        df_test_GS = df_test[ df_test['Series'] == 'GS']
    
        df_round_GS = pd.DataFrame(1-(df_bad_pred_GS['Round'].value_counts()/df_test_GS['Round'].value_counts()))
        df_round_GS = df_round_GS.reset_index().rename(columns = {'Round':'Pourcentage', 'index' : 'Round'})
            
        fig = px.bar(df_round_GS, x='Round', y='Pourcentage')
        st.plotly_chart(fig)
        
    elif variable == 'Tours Master 1000':
            
        df_bad_pred_M1000 = df_bad_pred[ df_bad_pred['Series'] == 'M1000']
        df_test_M1000 = df_test[ df_test['Series'] == 'M1000']
    
        df_round_M1000 = pd.DataFrame(1-(df_bad_pred_M1000['Round'].value_counts()/df_test_M1000['Round'].value_counts()))
        df_round_M1000 = df_round_M1000.reset_index().rename(columns = {'Round':'Pourcentage', 'index' : 'Round'})
        df_round_M1000.set_index('Round').transpose()[['1R','2R','3R','8th','Q','SF','F']].transpose().reset_index()
            
        fig = px.bar(df_round_M1000, x='Round', y='Pourcentage')
        st.plotly_chart(fig)
    
    elif variable == 'Meilleur tournois' :
                
        df_tournament = pd.DataFrame(1-(df_bad_pred['Tournament'].value_counts()/df_test['Tournament'].value_counts()))
        df_tournament = df_tournament.reset_index().rename(columns={'Tournament':'Pourcentage', 'index':'Tournament'})
        df_tournament_best = df_tournament.sort_values(by='Pourcentage', ascending = False).head(8)
            
        fig = px.bar(df_tournament_best, y='Tournament', x='Pourcentage', orientation='h')
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(fig)
            
    else :
        
        df_tournament = pd.DataFrame(1-(df_bad_pred['Tournament'].value_counts()/df_test['Tournament'].value_counts()))    
        df_tournament = df_tournament.reset_index().rename(columns={'Tournament':'Pourcentage', 'index':'Tournament'})
        df_tournament_bad = df_tournament.sort_values(by='Pourcentage', ascending = True).head(8)
            
        fig = px.bar(df_tournament_bad, y='Tournament', x='Pourcentage', orientation='h')
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(fig)
        
    st.markdown("* Le taux de prédictions réussies ne sont pas les mêmes d'une modalité à une autre. Dans le cas de la variable Series \
                on peut observer que le taux de réussite varie d'une catégorie de tournoi à une autre. Le taux en Grand Chelem est de 13% \
                supérieur à celui d'un ATP 250. Un facteur peut expliquer ce résultat. \
                On retrouve tous les types de joueurs en Grand Chelem, de la première à la 150 ème place donnant lieu à des matchs entre des \
                joueurs très bien classés et des joueurs très mal classés donc avec des résultats parfois joués d'avance. Tandis que dans \
                les ATP 250, très souvent on retrouve beaucoup de joueurs de seconde zone avec des niveaux très similaires laissant plus \
                de place à des suprises lors des matchs.")
        

with st.expander(label = 'Second modèle'):
    
    exec(open("7_2_modeles_ML.py").read())
    
    