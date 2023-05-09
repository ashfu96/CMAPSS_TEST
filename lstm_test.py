#######################################
           ### LIBRERIE ###
#######################################

# gestione di tipi dataframe 
import numpy as np
import pandas as pd

# Tensorflow 2
import tensorflow as tf
# keras
from tensorflow import keras
import keras.backend as K

# modelli sequenziali
from keras.models import Sequential, load_model
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, LSTM

# funzioni di sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

# lib per plot
import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn as sns

# accuracy e matrice di confusione
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# Setting del randomseed per la riproducibilità
np.random.seed(1234)  
PYTHONHASHSEED = 0

#######################################
           ### DATASET ###
#######################################

#url da github
url_TRAIN = "https://raw.githubusercontent.com/ashfu96/CMAPSS_TEST/main/train_FD001.txt"
url_TEST = "https://raw.githubusercontent.com/ashfu96/CMAPSS_TEST/main/test_FD001.txt"
url_RUL= "https://raw.githubusercontent.com/ashfu96/CMAPSS_TEST/main/RUL_FD001.txt"

df_train = pd.read_csv(url_TRAIN, sep=" ", header=None)
df_test = pd.read_csv(url_TEST, sep=" ", header=None)
df_rul = pd.read_csv(url_RUL, sep=" ", header=None)

train_copy = df_train
test_copy = df_test

#######################################
       ### MODIFICHE DATASET ###
#######################################

# Rimozione di colonne con NaN value
df_train.drop(columns=[26,27], axis=1, inplace=True)
df_test.drop(columns=[26,27], axis=1, inplace=True)
df_rul.drop(columns=[1], axis=1, inplace=True)

# Ridenominazione delle colonne con le label
columns_train = ['unit_ID','time_in_cycles','setting_1', 'setting_2','setting_3','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]

df_train.columns = columns_train
df_test.columns = columns_train

#######################################
    ### LABELING & NORMALIZATION ###
#######################################

# TRAIN

# Data Labeling - generazione della colonna RUL(Remaining Usefull Life)
rul = pd.DataFrame(df_train.groupby('unit_ID')['time_in_cycles'].max()).reset_index()
rul.columns = ['unit_ID', 'max']
df_train = df_train.merge(rul, on=['unit_ID'], how='left')
df_train['RUL'] = df_train['max'] - df_train['time_in_cycles']
df_train.drop('max', axis=1, inplace=True)

#  Normalizazione MinMax (from 0 to 1)
df_train['cycle_norm'] = df_train['time_in_cycles']
cols_normalize = df_train.columns.difference(['unit_ID','time_in_cycles','RUL'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(df_train[cols_normalize]), 
                             columns=cols_normalize, 
                             index=df_train.index)
join_df = df_train[df_train.columns.difference(cols_normalize)].join(norm_train_df)
df_train = join_df.reindex(columns = df_train.columns)

############################################

# TEST

# MinMax normalization (from 0 to 1)
df_test['cycle_norm'] = df_test['time_in_cycles']
cols_normalize_2 = df_test.columns.difference(['unit_ID','time_in_cycles','RUL'])
norm_test_df = pd.DataFrame(min_max_scaler.transform(df_test[cols_normalize_2]), 
                            columns=cols_normalize_2, 
                            index=df_test.index)
test_join_df = df_test[df_test.columns.difference(cols_normalize_2)].join(norm_test_df)
df_test = test_join_df.reindex(columns = df_test.columns)
df_test = df_test.reset_index(drop=True)

###########################################

# REAL RUL

true_rul = pd.read_csv(url_RUL, sep = '\s+', header = None)

# We use the ground truth dataset to generate labels for the test data.
# generate column max for test data
rul = pd.DataFrame(df_test.groupby('unit_ID')['time_in_cycles'].max()).reset_index()
rul.columns = ['unit_ID', 'max']
true_rul.columns = ['more']
true_rul['unit_ID'] = true_rul.index + 1
true_rul['max'] = rul['max'] + true_rul['more']
true_rul.drop('more', axis=1, inplace=True)

# generate RUL for test data
df_test = df_test.merge(true_rul, on=['unit_ID'], how='left')
df_test['RUL'] = df_test['max'] - df_test['time_in_cycles']
df_test.drop('max', axis=1, inplace=True)

#######################################
    ### PARAMETRI DIMENSIONALI ###
#######################################

n_train = df_train.shape[0]
n_test = df_test.shape[0]
df_train = df_train.assign(label1=[None]*n_train, label2=[None]*n_train)
df_test = df_test.assign(label1=[None]*n_test, label2=[None]*n_test)

# pick a large window size of 50 cycles
sequence_length = 50

def gen_sequence(id_df, seq_length, seq_cols):

    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]
        
# pick the feature columns 
sensor_cols = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
       'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
       'PCNfR_dmd', 'W31', 'W32']
sequence_cols = ['setting_1', 'setting_2', 'setting_3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# TODO for debug 
# val is a list of 192 - 50 = 142 bi-dimensional array (50 rows x 25 columns)
val=list(gen_sequence(df_train[df_train['unit_ID']==1], sequence_length, sequence_cols))

# generator for the sequences
# transform each id of the train dataset in a sequence
seq_gen = (list(gen_sequence(df_train[df_train['unit_ID']==id], sequence_length, sequence_cols)) 
           for id in df_train['unit_ID'].unique())

# generate sequences and convert to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
#print(seq_array.shape)

# function to generate labels
def gen_labels(id_df, seq_length, label):
    
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]

    return data_matrix[seq_length:num_elements, :]

# generate labels
label_gen = [gen_labels(df_train[df_train['unit_ID']==id], sequence_length, ['RUL']) 
             for id in df_train['unit_ID'].unique()]

label_array = np.concatenate(label_gen).astype(np.float32)
#label_array.shape

#######################################
    ### PARAMETRI DIMENSIONALI ###
#######################################

# define path to save model
model_path = 'regression_model.h5'

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Next, we build a deep network. 
# The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units. 
# Dropout is also applied after each LSTM layer to control overfitting. 
# Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()
model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=100,
         return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(units=30))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(Dense(units=20))
model.add(Activation("relu"))
model.add(Dense(units=nb_out))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer='nadam',metrics=['mae',r2_keras])

print(model.summary())

# fit the network
history = model.fit(seq_array, label_array, epochs=100, batch_size=50, validation_split=0.05, verbose=2,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )


#########################################
#########################################

import os

# We pick the last sequence for each id in the test data
seq_array_test_last = [df_test[df_test['unit_ID']==id][sequence_cols].values[-sequence_length:] 
                       for id in df_test['unit_ID'].unique() if len(df_test[df_test['unit_ID']==id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)


# Similarly, we pick the labels
y_mask = [len(df_test[df_test['unit_ID']==id]) >= sequence_length for id in df_test['unit_ID'].unique()]
label_array_test_last = df_test.groupby('unit_ID')['RUL'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)


# if best iteration's model was saved then load and use it
if os.path.isfile(model_path):
    estimator = load_model(model_path,custom_objects={'r2_keras': r2_keras})

    # test metrics
    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
    #print('\nMAE: {}'.format(scores_test[1]))
    #print('\nR^2: {}'.format(scores_test[2]))

    y_pred_test = estimator.predict(seq_array_test_last)
    y_true_test = label_array_test_last

    test_set = pd.DataFrame(y_pred_test)
    
#######################################
          ### STREAMLIT ###
#######################################
import streamlit as st

#st.title('Predizione della vita utile residua per singola unità')
#st.write("Predizione della vita utile residua per singola unità")

# Sidebar per la selezione dell'unità
#unit_id = st.sidebar.selectbox('Seleziona l\'unità da analizzare tramite :red[ID]:', list(df_test['unit_ID'].unique()))

#######################################################################
st.title('Predizione della vita utile residua per singola unità')
           
unit_id = st.sidebar.selectbox('Seleziona l\'unità da analizzare:', list(df_test['unit_ID'].unique()))

# Seleziona i dati relativi all'unità di interesse
seq_array_test_last_unit = seq_array_test_last[np.where(df_test['unit_ID'].unique() == unit_id)[0][0]].reshape(1, sequence_length, -1)
label_array_test_last_unit = label_array_test_last[np.where(df_test.groupby('unit_ID')['RUL'].nth(-1).values == df_test[df_test['unit_ID'] == unit_id]['RUL'].values[-1])[0][0]].reshape(1,1)

# Esegui la predizione e mostra i risultati
y_pred_test = estimator.predict(seq_array_test_last_unit)
y_true_test = label_array_test_last_unit

if y_pred_test < 75:
           st.markdown(" <font color='red'> ATTENZIONE! Il valore predetto per l'unità {} è: {}".format(unit_id, y_pred_test[0][0]))
else:
           st.write("Il valore predetto per l'unità {} è: {}".format(unit_id, y_pred_test[0][0]))

#st.write("Il valore reale per l'unità {} è: {}".format(unit_id, y_true_test[0][0]))
