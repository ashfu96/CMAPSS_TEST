# prova in streamlit per una dashboard di selezione unità
# e visualizzazione rul predetto (statico di prova)

# gestione di tipi dataframe 
import numpy as np
import pandas as pd
import random

# URL del dataset della repository
url = "https://raw.githubusercontent.com/ashfu96/CMAPSS_TEST/main/test_FD001.txt"
df = pd.read_csv(url, delimiter=" ")

#######################################################
############### COMANDI STREAMLIT #####################
#######################################################
import streamlit as st
# streamlit run prova.py

#titolo
st.title('Predizione del Rul')
st.write("Predizione della vita utile residua per singola unità")

#menu a tendina per selezione unit_ID
#opzioni = list(df["1"].unique())
#selezione = st.selectbox("Seleziona l'unità tramite ID", opzioni)
#df_filtrato = df[df["1"] == selezione]

# Sidebar per la selezione dell'unità
selezione = st.sidebar.selectbox("Seleziona l'unità tramite ID", list(df["1"].unique()))

# genero un valore casuale di RUL
import random
R = random.randint(0, 30)

st.write(f"Scelta l'unità {selezione} \n")

if R <= 10:
    st.markdown(f"<h1 style='color:red'> ATTENZIONE! L'unità {selezione} può effettuare altri {R} voli! .</h1>", unsafe_allow_html=True)
else:
    st.write(f" L'unità {selezione} può effettuare altri {R} voli")

