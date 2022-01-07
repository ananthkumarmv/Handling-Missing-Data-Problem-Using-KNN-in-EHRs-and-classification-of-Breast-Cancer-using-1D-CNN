import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.optimizers import Adam

cancer_dataset = pd.read_csv('../Dataset/dataset_with_missing_values.csv')


cancer_dataset.head()


cancer_dataset = cancer_dataset.drop(columns='id', axis=1)



cancer_dataset.shape


cancer_dataset.isnull().sum()


cancer_dataset = cancer_dataset.dropna(thresh=cancer_dataset.shape[1]-9, axis=0)


cancer_dataset.describe()

cancer_dataset.shape


cancer_dataset.replace({'diagnosis': {'B':0, 'M':1}}, inplace=True)

cancer_dataset.head()



cancer_dataset['diagnosis'].value_counts()






cancer_dataset.info()


cancer_dataset.groupby('diagnosis').mean()


correlation = cancer_dataset.corr()



correlation

data = cancer_dataset.values


data



data.shape[1]


ix = [i for i in range(data.shape[1]) if i != 0] 



x, y = data[:, ix], data[:, 0]




print('Missing: %d' % sum(np.isnan(x).flatten())) 
# Using KNNImputer to Impute Missing Values

# define imputer 
imputer = KNNImputer(n_neighbors=6)





#fit on the dataset
imputer.fit(x)





print('Missing: %d' % sum(np.isnan(xtrans).flatten()))



xtrans.shape


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(xtrans)
x = scaler.transform(xtrans)



from sklearn.decomposition import PCA


pca_test = PCA(n_components=30)
pca_test.fit(x)
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumilative variance")






# picking 5 components
n_PCA_components = 5
pca = PCA(n_components=n_PCA_components)
principalComponents = pca.fit_transform(x)






principalDf = pd.DataFrame(data=principalComponents, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])


principalDf.head()




print('Explained variance per principal component: {}'.format(pca.explained_variance_ratio_))


sum_list = [0.4426272,  0.18964872, 0.09390584, 0.06601566, 0.05492504]
sum(sum_list*100)




print('Amount of information lost due to PCA: ', (1-np.sum(pca.explained_variance_ratio_))*100, '%')



ydf = pd.DataFrame(y, columns=['y']) 




# finalDf = pd.concat([principalDf, cancer_dataset[['diagnosis']]], axis=1)
finalDf = pd.concat([principalDf, ydf], axis=1)



# finalDf = pd.concat([principalDf, cancer_dataset[['diagnosis']]], axis=1)
finalDf = pd.concat([principalDf, ydf], axis=1)





finalDf.shape




finalDf


final_x = finalDf.drop(columns = ['y'], axis=1)


final_y = finalDf['y']
x_train, x_test, y_train, y_test = train_test_split(final_x, y, test_size = 0.2, random_state = 3, stratify=y)








print(final_x.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)





scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(4089, 5, 1)
x_test = x_test.reshape(1023, 5, 1)




epochs = 3
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(5,1)))
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




model.summary()





model.compile(optimizer=Adam(learning_rate=0.001), loss = 'binary_crossentropy', metrics=['accuracy'])





history = model.fit(x_train, y_train, 
                    epochs=epochs, 
                    #callbacks=callbacks_list, 
                    validation_data = (x_test, y_test),
                    verbose=1)



print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()










import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import StandardScaler
#from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score 

from tkinter import *


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
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # 17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
    input_data = (e1.get(), e2.get(), e3.get(), e4.get(), e5.get(), e6.get(), e7.get(), e8.get(), e9.get(), e10.get(), e11.get(), e12.get(), e13.get(), e14.get(), e15.get(), e16.get(), e17.get(), e18.get(), e19.get(), e20.get(), e21.get(), e22.get(), e23.get(), e24.get(), e25.get(), e26.get(), e27.get(), e28.get(), e29.get(), e30.get())
    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshaping the numpy array
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
    # standardizing the data
    std_data = scaler.transform(input_data_reshape)
    prediction = model.predict(std_data)
    if(prediction[0]==0):
        myText.set("Benign")
    else:
        myText.set("Malignant")


        
master = Tk()
myText=StringVar()
master.title("Classification of Cancer")
# root.iconbitmap('Images/codemy.ico')
# master.configure(bg='BLUE')
master.geometry("650x800")



Label(master, text="Radius: ").grid(row=0, sticky=W+E)
Label(master, text="Texture: ").grid(row=1, sticky=W+E)
Label(master, text="Perimeter: ").grid(row=2, sticky=W+E)
Label(master, text="Area: ").grid(row=3, sticky=W+E)
Label(master, text="Smoothness: ").grid(row=4, sticky=W+E)
Label(master, text="Compactness: ").grid(row=5, sticky=W+E)
Label(master, text="Concavity: ").grid(row=6, sticky=W+E)
Label(master, text="Concave Points: ").grid(row=7, sticky=W+E)
Label(master, text="Symmetry: ").grid(row=8, sticky=W+E)
Label(master, text="Fractal Dimension: ").grid(row=9, sticky=W+E)
Label(master, text="Radius_se: ").grid(row=10, sticky=W+E)
Label(master, text="Texture_se: ").grid(row=11, sticky=W+E)
Label(master, text="Perimeter_se: ").grid(row=12, sticky=W+E)
Label(master, text="Area_se: ").grid(row=13, sticky=W+E)
Label(master, text="Smoothness_se: ").grid(row=14, sticky=W+E)
Label(master, text="Compactness_se: ").grid(row=15, sticky=W+E)
Label(master, text="Concavity_se: ").grid(row=16, sticky=W+E)
Label(master, text="Concave Points_se: ").grid(row=17, sticky=W+E)
Label(master, text="Symmetry_se: ").grid(row=18, sticky=W+E)
Label(master, text="Fractal Dimension_se: ").grid(row=19, sticky=W+E)
Label(master, text="Radius Worst: ").grid(row=20, sticky=W+E)
Label(master, text="Texture Worst: ").grid(row=21, sticky=W+E)
Label(master, text="Perimeter Worst: ").grid(row=22, sticky=W+E)
Label(master, text="Area Worst: ").grid(row=23, sticky=W+E)
Label(master, text="Smoothness Worst: ").grid(row=24, sticky=W+E)
Label(master, text="Compactness Worst: ").grid(row=25, sticky=W+E)
Label(master, text="Concavity Worst: ").grid(row=26, sticky=W+E)
Label(master, text="Concave Woints Worst: ").grid(row=27, sticky=W+E)
Label(master, text="Symmetry Worst: ").grid(row=28, sticky=W+E)
Label(master, text="Fractal Wimension Worst: ").grid(row=29, sticky=W+E)

Label(master, text="Result:").grid(row=35,column=0, sticky=W)


result=Label(master, text="", textvariable=myText).grid(row=35, column=1, sticky=W+E)




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


b = Button(master, text="Show Result", command=classify)
b.grid(row=30, column=5, columnspan=2, rowspan=2,sticky=W+E+N+S, padx=5, pady=5)


mainloop()




"source": [
    "history = model.fit(x_train, y_train, \n",
    "                    epochs=epochs, \n",
    "                    #callbacks=callbacks_list, \n",
    "                    validation_data = (x_test, y_test),\n",
    "                    verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa8ddd8f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "f1976350",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(history.history.keys())\n",
    "# summarize history for accuracy\n",
    "plt.plot(history.history['accuracy'])\n",
    "plt.plot(history.history['val_accuracy'])\n",
    "plt.title('model accuracy')\n",
    "plt.ylabel('accuracy')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'test'], loc='upper left')\n",
    "plt.show()\n",
    "# summarize history for loss\n",
    "plt.plot(history.history['loss'])\n",
    "plt.plot(history.history['val_loss'])\n",
    "plt.title('model loss')\n",
    "plt.ylabel('loss')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'test'], loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "794eb684",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4/4 [==============================] - 0s 4ms/step - loss: 0.3167 - accuracy: 0.9649\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[0.3166663646697998, 0.9649122953414917]"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.evaluate(x_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "fefd063b",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'method' object is not subscriptable",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-55-e42b694b2172>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      8\u001b[0m \u001b[1;31m#    print(\"M\")\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      9\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 10\u001b[1;33m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m25\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     11\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     12\u001b[0m \u001b[1;31m# print(model.predict(x_train[index].T))\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: 'method' object is not subscriptable"
     ]
    }
   ],
   "source": [
    "# for index in range(10):\n",
    "#     print(model.predict(x_train[25].T))\n",
    "\n",
    "    \n",
    "# if model.predict(x_train[25].T)[0][0] > 0.5:\n",
    "#    print(\"B\")\n",
    "# else:\n",
    "#    print(\"M\")\n",
    "\n",
    "print(model.predict[25])\n",
    "\n",
    "# print(model.predict(x_train[index].T))\n",
    "# print(y_train[28])\n",
    "# print(\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "c5df634d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(30, 1)"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train[0].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "ab43e21a",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_data = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "43468bac",
   "metadata": {},
   "outputs": [],
   "source": [
    "# changing input data to a numpy array\n",
    "input_data_as_numpy_array = np.asarray(input_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "6406384e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# reshaping the numpy array\n",
    "input_data_reshape = input_data_as_numpy_array.reshape(1, -1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "e70c6135",
   "metadata": {},
   "outputs": [],
   "source": [
    "# standardizing the data\n",
    "std_data = scaler.transform(input_data_reshape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "942d2ed3",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "in user code:\n\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\training.py:1586 predict_function  *\n        return step_function(self, iterator)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\training.py:1576 step_function  **\n        outputs = model.distribute_strategy.run(run_step, args=(data,))\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\tensorflow\\python\\distribute\\distribute_lib.py:1286 run\n        return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\tensorflow\\python\\distribute\\distribute_lib.py:2849 call_for_each_replica\n        return self._call_for_each_replica(fn, args, kwargs)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\tensorflow\\python\\distribute\\distribute_lib.py:3632 _call_for_each_replica\n        return fn(*args, **kwargs)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\training.py:1569 run_step  **\n        outputs = model.predict_step(data)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\training.py:1537 predict_step\n        return self(x, training=False)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\base_layer.py:1020 __call__\n        input_spec.assert_input_compatibility(self.input_spec, inputs, self.name)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\input_spec.py:229 assert_input_compatibility\n        raise ValueError('Input ' + str(input_index) + ' of layer ' +\n\n    ValueError: Input 0 of layer sequential is incompatible with the layer: : expected min_ndim=3, found ndim=2. Full shape received: (None, None)\n",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-48-b75db530bc87>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mprediction\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mstd_data\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\keras\\engine\\training.py\u001b[0m in \u001b[0;36mpredict\u001b[1;34m(self, x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)\u001b[0m\n\u001b[0;32m   1749\u001b[0m           \u001b[1;32mfor\u001b[0m \u001b[0mstep\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mdata_handler\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msteps\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1750\u001b[0m             \u001b[0mcallbacks\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mon_predict_batch_begin\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mstep\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1751\u001b[1;33m             \u001b[0mtmp_batch_outputs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1752\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mdata_handler\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshould_sync\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1753\u001b[0m               \u001b[0mcontext\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0masync_wait\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\tensorflow\\python\\eager\\def_function.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwds)\u001b[0m\n\u001b[0;32m    883\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    884\u001b[0m       \u001b[1;32mwith\u001b[0m \u001b[0mOptionalXlaContext\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_jit_compile\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 885\u001b[1;33m         \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    886\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    887\u001b[0m       \u001b[0mnew_tracing_count\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mexperimental_get_tracing_count\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\tensorflow\\python\\eager\\def_function.py\u001b[0m in \u001b[0;36m_call\u001b[1;34m(self, *args, **kwds)\u001b[0m\n\u001b[0;32m    922\u001b[0m       \u001b[1;31m# In this case we have not created variables on the first call. So we can\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    923\u001b[0m       \u001b[1;31m# run the first trace but we should fail if variables are created.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 924\u001b[1;33m       \u001b[0mresults\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_stateful_fn\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    925\u001b[0m       \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_created_variables\u001b[0m \u001b[1;32mand\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mALLOW_DYNAMIC_VARIABLE_CREATION\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    926\u001b[0m         raise ValueError(\"Creating variables on a non-first call to a function\"\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\tensorflow\\python\\eager\\function.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   3036\u001b[0m     \u001b[1;32mwith\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_lock\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3037\u001b[0m       (graph_function,\n\u001b[1;32m-> 3038\u001b[1;33m        filtered_flat_args) = self._maybe_define_function(args, kwargs)\n\u001b[0m\u001b[0;32m   3039\u001b[0m     return graph_function._call_flat(\n\u001b[0;32m   3040\u001b[0m         filtered_flat_args, captured_inputs=graph_function.captured_inputs)  # pylint: disable=protected-access\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\tensorflow\\python\\eager\\function.py\u001b[0m in \u001b[0;36m_maybe_define_function\u001b[1;34m(self, args, kwargs)\u001b[0m\n\u001b[0;32m   3457\u001b[0m               \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minput_signature\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m \u001b[1;32mand\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3458\u001b[0m               call_context_key in self._function_cache.missed):\n\u001b[1;32m-> 3459\u001b[1;33m             return self._define_function_with_shape_relaxation(\n\u001b[0m\u001b[0;32m   3460\u001b[0m                 args, kwargs, flat_args, filtered_flat_args, cache_key_context)\n\u001b[0;32m   3461\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\tensorflow\\python\\eager\\function.py\u001b[0m in \u001b[0;36m_define_function_with_shape_relaxation\u001b[1;34m(self, args, kwargs, flat_args, filtered_flat_args, cache_key_context)\u001b[0m\n\u001b[0;32m   3379\u001b[0m           expand_composites=True)\n\u001b[0;32m   3380\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 3381\u001b[1;33m     graph_function = self._create_graph_function(\n\u001b[0m\u001b[0;32m   3382\u001b[0m         args, kwargs, override_flat_arg_shapes=relaxed_arg_shapes)\n\u001b[0;32m   3383\u001b[0m     \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_function_cache\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marg_relaxed\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mrank_only_cache_key\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mgraph_function\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\tensorflow\\python\\eager\\function.py\u001b[0m in \u001b[0;36m_create_graph_function\u001b[1;34m(self, args, kwargs, override_flat_arg_shapes)\u001b[0m\n\u001b[0;32m   3296\u001b[0m     \u001b[0marg_names\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mbase_arg_names\u001b[0m \u001b[1;33m+\u001b[0m \u001b[0mmissing_arg_names\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3297\u001b[0m     graph_function = ConcreteFunction(\n\u001b[1;32m-> 3298\u001b[1;33m         func_graph_module.func_graph_from_py_func(\n\u001b[0m\u001b[0;32m   3299\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_name\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3300\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_python_function\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\func_graph.py\u001b[0m in \u001b[0;36mfunc_graph_from_py_func\u001b[1;34m(name, python_func, args, kwargs, signature, func_graph, autograph, autograph_options, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, override_flat_arg_shapes, acd_record_initial_resource_uses)\u001b[0m\n\u001b[0;32m   1005\u001b[0m         \u001b[0m_\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0moriginal_func\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtf_decorator\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0munwrap\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mpython_func\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1006\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1007\u001b[1;33m       \u001b[0mfunc_outputs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpython_func\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mfunc_args\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mfunc_kwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1008\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1009\u001b[0m       \u001b[1;31m# invariant: `func_outputs` contains only Tensors, CompositeTensors,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\tensorflow\\python\\eager\\def_function.py\u001b[0m in \u001b[0;36mwrapped_fn\u001b[1;34m(*args, **kwds)\u001b[0m\n\u001b[0;32m    666\u001b[0m         \u001b[1;31m# the function a weak reference to itself to avoid a reference cycle.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    667\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mOptionalXlaContext\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcompile_with_xla\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 668\u001b[1;33m           \u001b[0mout\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mweak_wrapped_fn\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__wrapped__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    669\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mout\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    670\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\func_graph.py\u001b[0m in \u001b[0;36mwrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    992\u001b[0m           \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m  \u001b[1;31m# pylint:disable=broad-except\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    993\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mhasattr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"ag_error_metadata\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 994\u001b[1;33m               \u001b[1;32mraise\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mag_error_metadata\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mto_exception\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    995\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    996\u001b[0m               \u001b[1;32mraise\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: in user code:\n\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\training.py:1586 predict_function  *\n        return step_function(self, iterator)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\training.py:1576 step_function  **\n        outputs = model.distribute_strategy.run(run_step, args=(data,))\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\tensorflow\\python\\distribute\\distribute_lib.py:1286 run\n        return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\tensorflow\\python\\distribute\\distribute_lib.py:2849 call_for_each_replica\n        return self._call_for_each_replica(fn, args, kwargs)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\tensorflow\\python\\distribute\\distribute_lib.py:3632 _call_for_each_replica\n        return fn(*args, **kwargs)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\training.py:1569 run_step  **\n        outputs = model.predict_step(data)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\training.py:1537 predict_step\n        return self(x, training=False)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\base_layer.py:1020 __call__\n        input_spec.assert_input_compatibility(self.input_spec, inputs, self.name)\n    C:\\Users\\Unbeknownstguy\\anaconda3\\lib\\site-packages\\keras\\engine\\input_spec.py:229 assert_input_compatibility\n        raise ValueError('Input ' + str(input_index) + ' of layer ' +\n\n    ValueError: Input 0 of layer sequential is incompatible with the layer: : expected min_ndim=3, found ndim=2. Full shape received: (None, None)\n"
     ]
    }
   ],
   "source": [
    "prediction = model.predict(std_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "52331050",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "7a0f6a3a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3\n",
      "dee\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-62-ad2146a57ae8>:58: RuntimeWarning: invalid value encountered in true_divide\n",
      "  x /= x.std ()\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 2880x180 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3\n",
      "dee\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 2880x180 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3\n",
      "dee\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 2880x180 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n",
      "dee\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-62-ad2146a57ae8>:93: RuntimeWarning: invalid value encountered in float_scalars\n",
      "  x /= x.std ()\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x3.46154 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n",
      "dee\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x45 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n",
      "dee\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x45 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n",
      "dee\n"
     ]
    },
    {
     "data": {
      "image/png": "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EQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEAiEQAAAACJRAAAAAAkEgEAAACQSAQAAABAIhEAAAAAiUQAAAAAJBIBAAAAkEgEAAAAQCIRAAAAAIlEAAAAACQSAQAAAJBIBAAAAEBbikQz8/yZ+ezM3DozV+9x/aUz85nN18dm5mnbmBcAAACA7ThwJJqZM6q3VJdVF1cvmZmLdw378+pZa62nVr9UXXvQeQEAAADYnm3cSfSM6ta11m1rrW9U11VX7Byw1vrYWuvuzeHHqwu2MC8AAAAAW7KNSHR+9YUdx7dvzh3Pz1R/sIV5AQAAANiSM7fwHrPHubXnwJkf7lgk+sHjvtnMVdVVVef00C0sDwAAAID9bONOoturJ+44vqC6Y/egmXlq9bbqirXWV4/3Zmuta9dal6y1LnlIZ29heQAAAADsZxuR6BPVRTPzlJk5q3pxdf3OATPzpOp91cvWWn+2hTkBAAAA2KIDP2621rpnZl5Vfag6o3r7WuvmmXnF5vpbqzdW31P9xsxU3bPWuuSgcwMAAACwHbPWnh8f9IDwiDl3/cA857CXAQAAAHBkfGS955N73byzjcfNAAAAAHiQE4kAAAAAEIkAAAAAEIkAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEIAAAAgEQiAAAAABKJAAAAAEgkAgAAACCRCAAAAIBEIgAAAAASiQAAAABIJAIAAAAgkQgAAACARCIAAAAAEokAAAAASCQCAAAAIJEI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\n",
      "text/plain": [
       "<Figure size 1440x1440 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import random\n",
    "from   tensorflow.keras.preprocessing.image import img_to_array, load_img\n",
    "\n",
    "# Let's define a new Model that will take an image as input, and will output\n",
    "# intermediate representations for all layers in the previous model after\n",
    "# the first.\n",
    "successive_outputs = [layer.output for layer in model.layers[1:]]\n",
    "\n",
    "#visualization_model = Model(img_input, successive_outputs)\n",
    "visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)\n",
    "\n",
    "# Let's prepare a random input image of a cat or dog from the training set.\n",
    "# cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]\n",
    "# dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]\n",
    "\n",
    "# img_path = random.choice(cat_img_files + dog_img_files)\n",
    "# img = load_img(img_path, target_size=(150, 150))  # this is a PIL image\n",
    "\n",
    "# x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)\n",
    "# x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)\n",
    "\n",
    "# Rescale by 1/255\n",
    "x = x_train[1].T\n",
    "\n",
    "# Let's run our image through our network, thus obtaining all\n",
    "# intermediate representations for this image.\n",
    "successive_feature_maps = visualization_model.predict(x)\n",
    "\n",
    "# These are the names of the layers, so can have them as part of our plot\n",
    "layer_names = [layer.name for layer in model.layers]\n",
    "\n",
    "# -----------------------------------------------------------------------\n",
    "# Now let's display our representations\n",
    "# -----------------------------------------------------------------------\n",
    "for layer_name, feature_map in zip(layer_names, successive_feature_maps):\n",
    "  print(len(feature_map.shape))\n",
    "  \n",
    "  if len(feature_map.shape) == 3:\n",
    "    print(\"dee\")\n",
    "    #-------------------------------------------\n",
    "    # Just do this for the conv / maxpool layers, not the fully-connected layers\n",
    "    #-------------------------------------------\n",
    "    n_features = feature_map.shape[-1]  # number of features in the feature map\n",
    "    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)\n",
    "    \n",
    "    # We will tile our images in this matrix\n",
    "    display_grid = np.zeros((size, size * n_features))\n",
    "    \n",
    "    #-------------------------------------------------\n",
    "    # Postprocess the feature to be visually palatable\n",
    "    #-------------------------------------------------\n",
    "    for i in range(n_features):\n",
    "      # x  = feature_map[0, :, :, i]\n",
    "      x  = feature_map[0, :, i]\n",
    "      # x  = feature_map[0, i]\n",
    "      x -= x.mean()\n",
    "      x /= x.std ()\n",
    "      x *=  64\n",
    "      x += 128\n",
    "      x  = np.clip(x, 0, 255).astype('uint8')\n",
    "      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid\n",
    "\n",
    "    #-----------------\n",
    "    # Display the grid\n",
    "    #-----------------\n",
    "\n",
    "    scale = 40. / n_features\n",
    "    plt.figure( figsize=(scale * n_features, scale) )\n",
    "    plt.title ( layer_name )\n",
    "    plt.grid  ( False )\n",
    "    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) \n",
    "    plt.show()\n",
    "  if len(feature_map.shape) == 2:\n",
    "    print(\"dee\")\n",
    "    #-------------------------------------------\n",
    "    # Just do this for the conv / maxpool layers, not the fully-connected layers\n",
    "    #-------------------------------------------\n",
    "    n_features = feature_map.shape[-1]  # number of features in the feature map\n",
    "    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)\n",
    "    \n",
    "    # We will tile our images in this matrix\n",
    "    display_grid = np.zeros((size, size * n_features))\n",
    "    \n",
    "    #-------------------------------------------------\n",
    "    # Postprocess the feature to be visually palatable\n",
    "    #-------------------------------------------------\n",
    "    for i in range(n_features):\n",
    "      # x  = feature_map[0, :, :, i]\n",
    "      # x  = feature_map[0, :, i]\n",
    "      x  = feature_map[0, i]\n",
    "      x -= x.mean()\n",
    "      x /= x.std ()\n",
    "      x *=  64\n",
    "      x += 128\n",
    "      x  = np.clip(x, 0, 255).astype('uint8')\n",
    "      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid\n",
    "\n",
    "    #-----------------\n",
    "    # Display the grid\n",
    "    #-----------------\n",
    "\n",
    "    \n",
    "    scale = 20. / n_features\n",
    "    plt.figure( figsize=(scale * n_features, scale) )\n",
    "    plt.title ( layer_name )\n",
    "    plt.grid  ( False )\n",
    "    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) \n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3a5416a5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "99e8326b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c65bf64",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "665ebc54",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}






