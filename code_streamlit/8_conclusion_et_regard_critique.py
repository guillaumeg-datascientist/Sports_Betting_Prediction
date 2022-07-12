import streamlit as st

st.title('Conclusion & Autocritique')


st.markdown("<style>.streamlit-expanderHeader {font-family:Rockwell; font-size: 26px; font-weight: bold}</style>",unsafe_allow_html=True)

#Dans la fonction expander, expanded = True permet de spécifier que l'on souhaite avoir le l'expander déjà déroulé

with st.expander(label = 'Conclusion', expanded = True): 
       
    st.markdown("- Nos **résultats différent** significativement d'un modèle à un autre.")
    
    st.markdown("- Cela s'explique notamment par le fait que le **modèle n°2** n'a **pas** vocation à prédire **100%** des matches.")
    
    st.markdown("- Toutefois, quel que soit notre modèle, **nos résultats sont souvent à la hauteur** de ceux obtenus par les **bookmakers** dont c'est la profession, ce qui est satisfaisant.")
                

    
with st.expander(label = 'Autocritique'): 
       
    st.markdown("- Tester **davantage de modèles** comme le Boosting, Bagging ou encore Voting Classifier ainsi que des **réseaux de neurones**.")
    
    st.markdown("- Essayer de **combiner nos modèles n°1 et n°2** de manière hybride.")    

    st.markdown("- **Pondérer nos scores** ainsi que ceux des bookmarkers à l'aide des **probabilités décimales** (pas simplement « 0 ou 1 » selon si la prédiction est correcte ou fausse)")

    

# streamlit run C:\Users\guiga\introduction.py
    