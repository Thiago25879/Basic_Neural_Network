import pip

from tensorflow import keras
import pandas as pd
from zipfile import ZipFile
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "jena_climate_2009_2016.csv"
df = pd.read_csv(csv_path)
del zip_file
df = df.drop('Date Time', axis=1)
cols = ['p','T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 'sh','H2OC', 'rho', 'wv', 'mwv', 'wd']
df.columns = cols

y = df.loc[2*72:,'T']
lagged_x = []
for lag in range(72,2*72,12):
    lagged = df.shift(lag)
    lagged.columns = [x + '.lag' + str(lag) for x in lagged.columns]
    lagged_x.append(lagged)
df = pd.concat(lagged_x, axis=1)
df = df.iloc[2*72:,:] #drop missing values due to lags
print(df)

# apply a min max scaler
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

# Fit a PCA with maximum number of components
mypca = PCA()
mypca.fit(df)

# Make a scree plot
plt.plot(mypca.explained_variance_ratio_)
plt.show()

mypca = PCA(10)
df = mypca.fit_transform(df)