import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.callbacks import EarlyStopping


st.title("CMAPSS Data Preprocessing")
st.write("Please upload your train, test and RUL data in txt format.")

def load_data(data):
    return pd.read_csv(data, delimiter=" ", header=None)


    
def preprocess_data(train_data, test_data, RUL_data):
    # Drop the last two columns from train and test data
    train_data = train_data.drop([26, 27], axis=1)
    test_data = test_data.drop([26, 27], axis=1)

    # Rename the columns
    column_names = ["unit_id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
    train_data.columns = column_names
    test_data.columns = column_names

    # Remove the header from the RUL data and rename the column
    # Remove the header from the RUL data and rename the column
    RUL_data = RUL_data.drop(0, axis=0)
    RUL_data['RUL'] = RUL_data[0]
    #RUL_data = RUL_data.drop(0, axis=1)
    RUL_data = RUL_data.dropna(axis=1)
    # Calculate RUL for train_data
    rul_train = pd.DataFrame(train_data.groupby("unit_id")["cycle"].max()).reset_index()
    rul_train.columns = ["unit_id", "max_cycles"]
    train_data = train_data.merge(rul_train, on=["unit_id"], how="left")
    train_data["RUL"] = train_data["max_cycles"] - train_data["cycle"]
    train_data = train_data.drop("max_cycles", axis=1)

    return train_data, test_data, RUL_data






train_data_file = st.file_uploader("Upload Train Data (txt)", type="txt")
if train_data_file is not None:
    train_data = load_data(train_data_file)
    st.write("Train Data:")
    st.write(train_data.shape)

test_data_file = st.file_uploader("Upload Test Data (txt)", type="txt")
if test_data_file is not None:
    test_data = load_data(test_data_file)
    st.write("Test Data:")
    st.write(test_data.shape)

RUL_data_file = st.file_uploader("Upload RUL Data (txt)", type="txt")
if RUL_data_file is not None:
    RUL_data = load_data(RUL_data_file)
    st.write("RUL Data:")
    st.write(RUL_data.shape)

if train_data_file is not None and test_data_file is not None and RUL_data_file is not None:
    train_data, test_data, RUL_data = preprocess_data(train_data, test_data, RUL_data)
    st.write("Preprocessed Train Data:")
    st.write(train_data.head())
    st.write("Preprocessed Test Data:")
    st.write(test_data.head())
    st.write("Preprocessed RUL Data:")
    st.write(RUL_data.head())
    

def create_X_y(data, seq_length):
    X = []
    y = []
    unit_id_to_indices = {}

    for unit_id in data['unit_id'].unique():
        unit_data = data[data['unit_id'] == unit_id]
        unit_indices = []
        for i in range(len(unit_data) - seq_length):
            X.append(unit_data.iloc[i : i + seq_length].drop('RUL', axis=1).values)
            y.append(unit_data.iloc[i + seq_length]['RUL'])
            unit_indices.append(len(X) - 1)
        unit_id_to_indices[unit_id] = unit_indices

    return X, y, unit_id_to_indices
from sklearn.model_selection import train_test_split

def train_val_split(unit_ids, unit_id_to_indices, test_size=0.2, random_state=42):
    train_unit_ids, val_unit_ids = train_test_split(unit_ids, test_size=test_size, random_state=random_state)
    
    train_indices = [idx for unit_id in train_unit_ids for idx in unit_id_to_indices[unit_id]]
    val_indices = [idx for unit_id in val_unit_ids for idx in unit_id_to_indices[unit_id]]
    
    return train_indices, val_indices

def create_train_val_arrays(X, y, train_indices, val_indices):
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_val = [X[i] for i in val_indices]
    y_val = [y[i] for i in val_indices]
    
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

def build_lstm_model(input_shape, num_lstm_layers, activation_function, optimizer, weight_initializer, regularization_l1, regularization_l2, layer_normalization, batch_normalization):
    model = Sequential()
    regularizer = L1L2(l1=regularization_l1, l2=regularization_l2)

    for i in range(num_lstm_layers):
        return_sequences = i != num_lstm_layers - 1
        model.add(LSTM(units=50, activation=activation_function, kernel_initializer=weight_initializer, kernel_regularizer=regularizer, return_sequences=return_sequences))

        if layer_normalization:
            model.add(LayerNormalization())

        if batch_normalization:
            model.add(BatchNormalization())

    model.add(Dense(1))

    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stop])

    # Save the MAE
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    
    # Generate predictions on the validation data
    y_pred = model.predict(X_val).flatten()

    return mae, val_mae, y_pred

# Add this line after your existing file uploader widgets
create_sequences_button = st.button("Create Sequences and Labels")
# Add the slider for sequence length selection after the file uploader widgets
seq_length = st.slider("Select Sequence Length", min_value=1, max_value=100, value=50, step=1)


# Add this after you create the sequences and labels
if create_sequences_button and train_data_file is not None:
    X, y, unit_id_to_indices = create_X_y(train_data, seq_length)
    

    # Add a selectbox for the user to choose the unit_id
    unique_unit_ids = train_data["unit_id"].unique()
    # Split the sequences into train and validation sets
    train_indices, val_indices = train_val_split(unique_unit_ids, unit_id_to_indices)
    
    

    # Create train and validation arrays
    X_train, y_train, X_val, y_val = create_train_val_arrays(X, y, train_indices, val_indices)

    
    st.write("Train sequences:")
    st.write(X_train.shape)
    st.write("Train labels:")
    st.write(y_train.shape)
    st.write("Validation sequences:")
    st.write(X_val.shape)
    st.write("Validation labels:")
    st.write(y_val.shape)
    
    

# Build and display the model
build_model_button = st.button("set_model")
    
if build_model_button is not None:
    # Create train and validation arrays
    # Model parameters
    num_lstm_layers = st.sidebar.slider("Number of LSTM layers", min_value=1, max_value=2, value=1, step=1)
    activation_function = st.sidebar.selectbox("Activation function", options=["tanh", "relu"])
    optimizer = st.sidebar.selectbox("Optimizer", options=["RMSprop", "adam"])
    weight_initializer = st.sidebar.selectbox("Weight initializer", options=["glorot_uniform", "he_uniform"])
    regularization_l1 = st.sidebar.number_input("L1 regularization", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    regularization_l2 = st.sidebar.number_input("L2 regularization", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    layer_normalization = st.sidebar.checkbox("Layer normalization")
    batch_normalization = st.sidebar.checkbox("Batch normalization")
    print("model set successfully!")
    # Build and display the model
    model = build_lstm_model(28, num_lstm_layers, activation_function, optimizer, weight_initializer, regularization_l1, regularization_l2, layer_normalization, batch_normalization)
    print("model bult successfully!")



# Add a button for training the model
train_model_button = st.button("Train Model")

if train_model_button:
    epochs = st.number_input("Number of Epochs", min_value=1, max_value=1000, value=10, step=1)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=32, step=1)
    mae, state_3.val_mae, y_pred = train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)

    # Display the MAE
    st.write("Training MAE:")
    st.write(mae[-1])
    st.write("Validation MAE:")
    st.write(val_mae[-1])

    # Display the plot of validation predictions versus real RUL values
    fig, ax = plt.subplots()
    ax.scatter(y_val, y_pred)
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
    ax.set_xlabel('Real RUL')
    ax.set_ylabel('Predicted RUL')
    st.pyplot(fig)
