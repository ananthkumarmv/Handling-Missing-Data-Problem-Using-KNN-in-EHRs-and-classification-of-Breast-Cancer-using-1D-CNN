import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score 



root = Tk()
root.title("Title goes here")
root.iconbitmap('Images/codemy.ico')
root.geometry("700x700")


def classify():
    cancer_dataset = pd.read_csv('../Dataset/dataset.csv')
    cancer_dataset = cancer_dataset.drop(columns='id', axis=1)
    cancer_dataset.replace({'diagnosis': {'B':0, 'M':1}}, inplace=True)
    x = cancer_dataset.drop(columns='diagnosis', axis=1)
    y = cancer_dataset['diagnosis']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    model = LogisticRegression()
    # training the Logistic Regression Model with Training Data
    model.fit(x_train, y_train)
    # 17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
    input_data = ()
    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshaping the numpy array
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
    # standardizing the data
    std_data = scaler.transform(input_data_reshape)
    prediction = model.predict(std_data)
    if(prediction[0]==0):
        print("Benign")
    else:
        print("Malignant")



master = Tk()
myText=StringVar()



Label(master, text="radius_mean").grid(row=0, sticky=W)
Label(master, text="texture_mean").grid(row=1, sticky=W)
Label(master, text="perimeter_mean").grid(row=2, sticky=W)
Label(master, text="area_mean").grid(row=3, sticky=W)
Label(master, text="smoothness_mean").grid(row=4, sticky=W)
Label(master, text="compactness_mean").grid(row=5, sticky=W)
Label(master, text="concavity_mean").grid(row=6, sticky=W)
Label(master, text="concave_points_mean").grid(row=7, sticky=W)
Label(master, text="symmetry_mean").grid(row=8, sticky=W)
Label(master, text="fractal_dimension_mean").grid(row=9, sticky=W)
Label(master, text="radius_se").grid(row=10, sticky=W)
Label(master, text="texture_se").grid(row=11, sticky=W)
Label(master, text="perimeter_se").grid(row=12, sticky=W)
Label(master, text="area_se").grid(row=13, sticky=W)
Label(master, text="smoothness_se").grid(row=14, sticky=W)
Label(master, text="compactness_se").grid(row=15, sticky=W)
Label(master, text="concavity_se").grid(row=16, sticky=W)
Label(master, text="concave points_se").grid(row=17, sticky=W)
Label(master, text="symmetry_se").grid(row=18, sticky=W)
Label(master, text="fractal_dimension_se").grid(row=19, sticky=W)
Label(master, text="radius_worst").grid(row=20, sticky=W)
Label(master, text="texture_worst").grid(row=21, sticky=W)
Label(master, text="perimeter_worst").grid(row=22, sticky=W)
Label(master, text="area_worst").grid(row=23, sticky=W)
Label(master, text="smoothness_worst").grid(row=24, sticky=W)
Label(master, text="compactness_worst").grid(row=25, sticky=W)
Label(master, text="concavity_worst").grid(row=26, sticky=W)
Label(master, text="concave points_worst").grid(row=27, sticky=W)
Label(master, text="First").grid(row=28, sticky=W)
Label(master, text="symmetry_worst").grid(row=29, sticky=W)
Label(master, text="fractal_dimension_worst").grid(row=30, sticky=W)

Label(master, text="Result:").grid(row=31, sticky=W)
result=Label(master, text="", textvariable=myText).grid(row=3,column=1, sticky=W)






e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)
e14 = Entry(master)
e15 = Entry(master)
e16 = Entry(master)
e17 = Entry(master)
e18 = Entry(master)
e19 = Entry(master)
e20 = Entry(master)
e21 = Entry(master)
e22 = Entry(master)
e23 = Entry(master)
e24 = Entry(master)
e25 = Entry(master)
e26 = Entry(master)
e27 = Entry(master)
e28 = Entry(master)
e29 = Entry(master)
e30 = Entry(master)



e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e5.grid(row=4, column=1)
e6.grid(row=5, column=1)
e7.grid(row=6, column=1)
e8.grid(row=7, column=1)
e9.grid(row=8, column=1)
e10.grid(row=9, column=1)
e11.grid(row=10, column=1)
e12.grid(row=11, column=1)
e13.grid(row=12, column=1)
e14.grid(row=13, column=1)
e15.grid(row=14, column=1)
e16.grid(row=15, column=1)
e17.grid(row=16, column=1)
e18.grid(row=17, column=1)
e19.grid(row=18, column=1)
e20.grid(row=19, column=1)
e21.grid(row=20, column=1)
e22.grid(row=21, column=1)
e23.grid(row=22, column=1)
e24.grid(row=23, column=1)
e25.grid(row=24, column=1)
e26.grid(row=25, column=1)
e27.grid(row=26, column=1)
e28.grid(row=27, column=1)
e29.grid(row=28, column=1)
e30.grid(row=29, column=1)

        

