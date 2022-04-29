import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from multiprocessing.sharedctypes import Value
from operator import index
import streamlit as st

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.optimizers import Adam

header = st.container()

st.markdown(
    """
    <style>
    .main{
    background-color: #000000
    }
    </style>
    """,
    unsafe_allow_html = True
)

@st.cache
def get_data(filename):
    dataset = pd.read_csv(filename)

    return dataset

# KNN Imputer
def KNNI(x):

    # define imputer 
    imputer = KNNImputer(n_neighbors=6)

    #fit on the dataset
    imputer.fit(x)

    # transforming the dataset and returning
    return imputer.transform(x)

# Performing categorical encoding
def categorical_encoding(dataset):
    dataset.replace({'diagnosis': {'B':0, 'M':1}}, inplace=True)

    return dataset

# Setting 70% as threshold
def set_seventy(dataset):
    dataset = dataset.dropna(thresh=cancer_dataset.shape[1]-9, axis=0)

    return dataset


with header:
    cancer_dataset = get_data(r'C:\Users\Unbeknownstguy\Documents\GitHub\Projects\Machine_Learning\Handling-Missing-Data-Problem-Using-KNN-in-EHRs-for-Cancer-Prediction\Dataset\data.csv')

    cancer_dataset = cancer_dataset.drop(columns='Unnamed: 32', axis=1)

    st.write(cancer_dataset.head())

    cancer_dataset = cancer_dataset.drop(columns='id', axis=1)

    st.write("Shape of the dataset: ")
    st.write(cancer_dataset.shape)

    st.write(cancer_dataset.isnull().sum())

    cancer_dataset = set_seventy(cancer_dataset)

    st.write("Shape of the dataset: ")
    st.write(cancer_dataset.shape)

    cancer_dataset = categorical_encoding(cancer_dataset)

    st.write(cancer_dataset['diagnosis'].value_counts())

    st.write(cancer_dataset.groupby('diagnosis').mean())

    data = cancer_dataset.values

    ix = [i for i in range(data.shape[1]) if i != 0] 

    x, y = data[:, ix], data[:, 0]

    st.write("Total number of missing values: ")
    st.write(sum(np.isnan(x).flatten()))

    xtrans = KNNI(x)

    st.write("Total number of missing values After using KNN Imputer: ")
    st.write(sum(np.isnan(x).flatten()))

    X_k = pd.DataFrame(xtrans, columns = [ 'radius_mean', 'texture_mean', 'perimeter_mean',
        'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
        'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst'])

    Y_k = pd.DataFrame(y, columns = ['diagnosis'])

    bestfeatures = SelectKBest(score_func=chi2, k=14)
    fit = bestfeatures.fit(X_k,Y_k)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_k.columns)

    #concatinating two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']


    st.write(featureScores.nlargest(14,'Score'))

    columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'radius_se', 'perimeter_se', 'area_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'compactness_worst', 'concavity_worst']

    X_new = pd.DataFrame(cancer_dataset, columns=columns)

    st.write(X_new.describe())

    x_train, x_test, y_train, y_test = train_test_split(X_new, Y_k, test_size = 0.2, random_state = 3, stratify=Y_k)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = x_train.reshape(455, 14, 1)
    x_test = x_test.reshape(114, 14, 1)

    epochs = 3
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(14,1)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    model.add(Conv1D(filters=16, kernel_size=2, activation='relu',))
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu',))
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu',))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(1, activation='sigmoid'))


    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='acc', patience=1)
    ]
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

    BATCH_SIZE = 400
    EPOCHS = 50

    model.compile(optimizer=Adam(learning_rate=0.01), loss = 'binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        #callbacks=callbacks_list, 
                        validation_data = (x_test, y_test),
                        verbose=1)


    st.write(model.evaluate(x_test, y_test))


    col1, col2 = st.columns(2)

    radius_mean = col1.text_input('Radius Mean')
    try:
        if float(radius_mean) in range(6.9810, 28.1100):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    texture_mean = col1.text_input('Texture Mean')
    try:
        if float(texture_mean) in range(9.7100, 39.2800):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    perimeter_mean = col1.text_input('Perimeter Mean')
    try:
        if float(perimeter_mean) in range(43.7900, 188.5000):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    area_mean = col1.text_input('Area Mean')
    try:
        if float(area_mean) in range(143.5000, 2,501.0000):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    concavity_mean = col1.text_input('Concavity Mean')
    try:
        if float(concavity_mean) in range(0.0000, 0.4268):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    radius_se = col1.text_input('Radius Se')
    try:
        if float(radius_se) in range(0.1115, 2.8730):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    perimeter_se = col1.text_input('Perimeter Se')
    try:
        if float(perimeter_se) in range(0.7570, 21.9800):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    area_se = col2.text_input('Area Se')
    try:
        if float(area_se) in range(6.8020, 542.2000):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    radius_worst = col2.text_input('Radius Worst')
    try:
        if float(radius_worst) in range(7.9300, 36.0400):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    texture_worst = col2.text_input('Texture Worts')
    try:
        if float(texture_worst) in range(12.0200, 49.5400):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    perimeter_worst = col2.text_input('Perimeter worst')
    try:
        if float(perimeter_worst) in range(50.4100, 251.2000):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    area_worst = col2.text_input('Area Wrost')
    try:
        if float(area_worst) in range(185.2000, 4254.0000):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    compactness_worst = col2.text_input('Compactness Wrost')
    try:
        if float(compactness_worst) in range(0.0273, 1.0580):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    concavity_worst = col2.text_input('Concavity Worst')
    try:
        if float(concavity_worst) in range(0.0000, 1.2520):
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    # code for prediction
    diagnosis = ''

    # creating a button for prediction

    if st.button('Result'):

        input_data = (radius_mean, texture_mean, perimeter_mean, area_mean, concavity_mean, radius_se, perimeter_se, area_se, radius_worst, texture_worst, perimeter_worst, area_worst, compactness_worst, concavity_worst)

        input_data_as_numpy_array = np.asarray(input_data)

        input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

        std_data = scaler.transform(input_data_reshape)

        prediction = model.predict(std_data)

        if prediction[0]<=0.5:
            diagnosis = 'Benign'
        else:
            diagnosis = 'Malignant'

    st.success(diagnosis)


