import streamlit as st

st.title('Introduction')

st.markdown("<style>.streamlit-expanderHeader {font-family:Rockwell; font-size: 26px; font-weight: bold}</style>",unsafe_allow_html=True)

#Dans la fonction expander, expanded = True permet de spécifier que l'on souhaite avoir le l'expander déjà déroulé

with st.expander(label = 'Problématiques', expanded=True): 
       
    st.markdown("- **Prédire** l’issue des **matches de tennis** masculins professionnels (ATP) des **20 dernières années**.")
                
    st.markdown("- **Enrichir** notre **dataset d’origine** (pauvre en données utiles) avec des **features pertinentes** théoriquement corrélées au vainqueur de chaque rencontre.")
                
    st.markdown("- Faire en sorte de ne **jamais** communiquer d’**informations en provenance du futur** à nos algorithmes.")
                
    st.markdown("- **Comparer** nos résultats avec ceux qu’auraient obtenus les **bookmakers** sur la même période.")
    
    st.markdown(" ")
    
    
    
with st.expander(label = 'Enjeux'): 
       
    st.markdown("- Rendre notre **dataset** d’origine « **machine-learning-friendly** »")
    
    st.markdown("- Faire en sorte que les **nouvelles features** créées soient **fiables**.")    

    st.markdown("- Obtenir les **meilleures prédictions** possibles.")   
    
    st.markdown(" ")
    
# streamlit run C:\Users\guiga\introduction.py