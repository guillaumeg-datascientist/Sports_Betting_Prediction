import streamlit as st
import pandas as pd
from PIL import Image
import io

st.title('Pipeline de préparation des données')

st.markdown("<style>.streamlit-expanderHeader {font-family:Rockwell; font-size: 26px; font-weight: bold}</style>",unsafe_allow_html=True)

#Dans la fonction expander, expanded = True permet de spécifier que l'on souhaite avoir le l'expander déjà déroulé

with st.expander(label = 'Initialisation des variables', expanded = True): 
    
    exec(open("6_2_pipeline.py").read())
    
#Ici on ne veut pas que l'expander soit déjà déroulé
    
with st.expander(label = 'Création des variables « player_A », « player_B », et « target »'): 
    
    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">1. Abstraction des notions de « Winner » et « Loser » au profit d’une variable « target » binaire et des variables « player_A » et « player_B »</p>'
    st.markdown(text, unsafe_allow_html=True) 
    
    code = '''# On crée la variable « target » avec des 1 et des 2 au hasard
df["target"] = np.random.choice([0, 1], df.shape[0])

# On crée les variables « player_A » et « player_B » de manière 100% aléatoire
df["player_A"] = np.where(df['target'] == 1, df["Winner"], df["Loser"])
df["player_B"] = np.where(df['player_A'] == df["Winner"], df["Loser"], df["Winner"])'''

    st.code(code, language='python')
    

        
    st.markdown(" \n")
    st.markdown(" \n")

    
    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">2. Ajout des 3 nouvelles variables au dataset</p>'
    st.markdown(text, unsafe_allow_html=True) 
    
    post_player = pd.read_csv("datasets_for_streamlit/post player a & b plus target.csv", sep = ";")
    st.dataframe(post_player)
    
    
    
with st.expander(label = 'Création des variables de type « head-to-head »'): 
    
    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">1. Identification d’un head-to-head unique (id_h2h), quel que soit les valeurs prises par les variables « player_A » et « player_B »</p>'
    st.markdown(text, unsafe_allow_html=True) 
    
    code = '''for i in range (0,58664):
    df1 = dfh.iloc[i:(i+1),:]
    player_A = df1.iloc[0]["player_A"]
    player_B = df1.iloc[0]["player_B"]
    list_a_and_b = [player_A, player_B]
    list_a_and_b.sort()
    id_h2h = str(list_a_and_b[0]) + " - " + str(list_a_and_b[1])
    id_h2h_list.append(id_h2h)
    
dfh["id_h2h"] = id_h2h_list   '''

    st.code(code, language='python')
    
    
        
    st.markdown(" \n")
    st.markdown(" \n")


    
    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">2. Identification des joueurs 1 et 2 de chaque id_h2h, de leurs nombres de victoires et du n° de la confrontation</p>'
    st.markdown(text, unsafe_allow_html=True) 
    
    st.markdown("*Exemple de l'id_h2h 'Federer R. - Nadal R.'*")
    df_rf_nadal = pd.read_csv("datasets_for_streamlit/df_rf_nadal.csv", sep = ";")
    st.dataframe(df_rf_nadal)
    

    
        
    st.markdown(" \n")
    st.markdown(" \n")


    
    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">3. Vérification</p>'
    st.markdown(text, unsafe_allow_html=True) 
    
    st.markdown("*https://www.atptour.com/en/players/atp-head-2-head/roger-federer-vs-rafael-nadal/f324/n409*")
    image = Image.open('images_for_streamlit/rfnadal.png')
    st.image(image, caption='Affichage des head-to-head sur le site officiel de l\'ATP')
    
    
    
    

    
with st.expander(label = 'Création du dataset final (« df.csv ») utilisable par les modèles de prédiction'): 
    
    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">1. Rassemblement des paires de variables</p>'
    st.markdown(text, unsafe_allow_html=True) 
    
    predf = pd.read_csv("datasets_for_streamlit/predf.csv")
    
    buffer = io.StringIO()
    predf.info(buf=buffer)
    s = buffer.getvalue()
    
    st.text(s)   
    
        
    st.markdown(" \n")
    st.markdown(" \n")


    
    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">2. Transformation de chaque « paire de variables » en une variable unique représentant au mieux l’écart entre ces deux valeurs</p>'
    st.markdown(text, unsafe_allow_html=True) 
    
    code = '''df["1_year_pts_delta"] = df["A_1_year_pts"] - df["B_1_year_pts"]
df["8_months_pts_delta"] = df["A_8_months_pts"] - df["B_8_months_pts"]
df["4_months_pts_delta"] = df["A_4_months_pts"] - df["B_4_months_pts"]
df["1_year_pts_delta_surface"] = df["A_1_year_pts_surface"] - df["B_1_year_pts_surface"]
df["8_months_pts_delta_surface"] = df["A_8_months_pts_surface"] - df["B_8_months_pts_surface"]
df["4_months_pts_delta_surface"] = df["A_4_months_pts_surface"] - df["B_4_months_pts_surface"]
df["h2h_player_score_delta"] = df["h2h_player_A_score"] - df["h2h_player_B_score"]
df["h2h_player_score_last_5_delta"] = df["h2h_player_A_score_last_5"] - df["h2h_player_B_score_last_5"]
df["h2h_player_score_surface_delta"] = df["h2h_player_A_score_surface"] - df["h2h_player_B_score_surface"]
df["h2h_player_score_last_5_surface_delta"] = df["h2h_player_A_score_last_5_surface"] - df["h2h_player_B_score_last_5_surface"]
df["days_since_last_match_played_delta"] = df["days_since_last_match_played_A"] - df["days_since_last_match_played_B"]
df["V-D_rang_delta"] = df["v-d_rang_A"] - df["v-d_rang_B"]
df["matchs_played_3w_delta"] = df["matchs_played_3w_A"] - df["matchs_played_3w_B"]
df["V-D_3w_delta"] = df["V-D_3w_A"] - df["V-D_3w_B"]
df["ratioV_3w_delta"] = df["ratioV_3w_A"] / df["ratioV_3w_B"]
df["V-D_3y_delta"] = df["V-D_3y_A"] - df["V-D_3y_B"]
df["ratioV_3y_delta"] = df["ratioV_3y_A"] / df["ratioV_3y_B"]
df["V-D_3y_format_delta"] = df["V-D_3y_format_A"] - df["V-D_3y_format_B"]
df["ratioV_3y_format_delta"] = df["ratioV_3y_format_A"] / df["ratioV_3y_format_B"]
df["V-D_3y_surface_delta"] = df["V-D_3y_surface_A"] - df["V-D_3y_surface_B"]
df["ratioV_3y_surface_delta"] = df["ratioV_3y_surface_A"] / df["ratioV_3y_surface_B"]
df["V-D_Series_and_Round_5y_delta"] = df["V-D_Series_and_Round_5y_A"] - df["V-D_Series_and_Round_5y_B"]
df["ratioV_Series_and_Round_5y_delta"] = df["ratioV_Series_and_Round_5y_A"] / df["ratioV_Series_and_Round_5y_B"]
df["V-D_tournament_5y_delta"] = df["V-D_tournament_5y_A"] - df["V-D_tournament_5y_B"]
df["ratioV_tournament_5y_delta"] = df["ratioV_tournament_5y_A"] / df["ratioV_tournament_5y_B"]'''

    st.code(code, language='python')

    
        
    st.markdown(" \n")
    st.markdown(" \n")


    
    text = '<p style="font-family:Helvetica; color:Black; font-size: 20px; font-weight: 600;">3. Dataset prêt à être utilisé par les modèles de machine learning</p>'
    st.markdown(text, unsafe_allow_html=True) 
    
    df = pd.read_csv("datasets_for_streamlit/df.csv", sep = ",")
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    
    st.text(s)     
    
#Ici on ne veut pas que l'expander soit déjà déroulé    

    
    # streamlit run C:\Users\guiga\pipeline.py

