# prova in streamlit per una dashboard di selezione unità
# e visualizzazione rul predetto (statico di prova)

# gestione di tipi dataframe 
import numpy as np
import pandas as pd
import random
import requests
from io import StringIO

# URL del dataset della repository
url = "https://raw.githubusercontent.com/ashfu96/CMAPSS_TEST/main/test_FD001.txt"

# Carica il file dal tuo repository Github
response = requests.get(url)
testo_file = response.text

# Converte il testo del file in un oggetto pandas DataFrame
df = pd.read_csv(StringIO(testo_file), delimiter=",")

#######################################################
############### COMANDI STREAMLIT #####################
#######################################################

streamlit run prova.ipynb

#titolo
st.title('Predizione del Rul')
st.write("Predizione della vita utile residua per singola unità")

#menu a tendina per selezione unit_ID
opzioni = list(df["unit_ID"].unique())
selezione = st.selectbox("Seleziona l'unità tramite ID", opzioni)
df_filtrato = df[df["unit_ID"] == selezione]

# genero un valore casuale di RUL
import random
R = random.randint(0, 30)

st.write(f"Scelta l'unità {selezione} \n")

if R <= 10:
    st.markdown(f"<h1 style='color:red'> ATTENZIONE! L'unità {selezione} può effettuare altri {R} voli! .</h1>", unsafe_allow_html=True)
else:
    st.write(f" L'unità {selezione} può effettuare altri {R} voli")

