############################ Building ANN Model  ##############################

#//// Importing Libraries Required
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

#//// Importing Dataset into the spyder
df = pd.read_csv("C:/Users/Welcome/Desktop/DATA SCIENCE/Tasks/ANN/Assignments/Datasets_ANN Assignment/fireforests.csv")

df.info()

description = df.describe()
#//// Checking presence of null values
df.isnull().sum()

#//// Checking for duplicate entries
dup = df.duplicated()
sum(dup)
df = df.drop_duplicates()

df = df.iloc[:, :11]

######################### Exploratory Data Analysis ###########################

#//// Measures of Central Tendency
mean = df.mean()
median = df.median()
mode = df.mode()

#//// Measures of Dispersion
variance = df.var()
std_dev = df.std()
range = df.iloc[:,2:].max() - df.iloc[:, 2:].min()

#//// Skewness
skewness = df.skew()

#//// Kurtosis
kurtosis = df.kurt()

df.columns


#//// Data visualization
#//// Univariate Analysis
import matplotlib.pyplot as plt
import seaborn as sbn

plt.bar(height = df.FFMC, x = np.arange(0,509,1))
plt.hist(df['FFMC'])
plt.boxplot(df['FFMC']) #//// Outliers Exist

plt.bar(height = df.DMC, x = np.arange(0,509,1))
plt.hist(df['DMC'])
plt.boxplot(df['DMC']) #//// Outliers Exist

plt.bar(height = df.DC, x = np.arange(0,509,1))
plt.hist(df['DC'])
plt.boxplot(df['DC']) #//// Outliers Exist

plt.bar(height = df.ISI, x = np.arange(0,509,1))
plt.hist(df['ISI'])
plt.boxplot(df['ISI']) #//// Outliers Exist

plt.bar(height = df.temp, x = np.arange(0,509,1))
plt.hist(df['temp'])
plt.boxplot(df['temp']) #//// Outliers Exist

plt.bar(height = df.RH, x = np.arange(0,509,1))
plt.hist(df['RH'])
plt.boxplot(df['RH']) #//// Outliers Exist

plt.bar(height = df.wind, x = np.arange(0,509,1))
plt.hist(df['wind'])
plt.boxplot(df['wind']) #//// Outliers Exist

plt.bar(height = df.rain, x = np.arange(0,509,1))
plt.hist(df['rain'])
plt.boxplot(df['rain']) #//// Outliers Exist

plt.bar(height = df.area, x = np.arange(0,509,1))
plt.hist(df['area'])
plt.boxplot(df['area']) #//// Outliers Exist

#//// Bivariate Analysis
plt.scatter(df['FFMC'], df['DMC'])
plt.scatter(df['DMC'], df['DC'])
plt.scatter(df['DC'], df['ISI'])
plt.scatter(df['ISI'], df['temp'])
plt.scatter(df['RH'], df['wind'])
plt.scatter(df['wind'], df['rain'])
plt.scatter(df['area'], df['FFMC'])
plt.scatter(df['FFMC'], df['DMC'])
###############################################################################

#//// Removing outliers
from feature_engine.outliers import Winsorizer

win_FFMC = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['FFMC'])
df['FFMC'] = win_FFMC.fit_transform(df[['FFMC']])

win_DC = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['DC'])
df['DC'] = win_DC.fit_transform(df[['DC']])

win_DMC = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['DMC'])
df['DMC'] = win_DMC.fit_transform(df[['DMC']])

win_ISI = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['ISI'])
df['ISI'] = win_ISI.fit_transform(df[['ISI']])

win_temp = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['temp'])
df['temp'] = win_temp.fit_transform(df[['temp']])

win_RH = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['RH'])
df['RH'] = win_RH.fit_transform(df[['RH']])

win_wind = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['wind'])
df['wind'] = win_wind.fit_transform(df[['wind']])

#win_rain = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['rain'])
#df['rain'] = win_rain.fit_transform(df[['rain']])

#win_area = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['area'])
#df['area'] = win_area.fit_transform(df[['area']])


#//// Applying Encoding on Categorical variables
from sklearn.preprocessing import LabelEncoder

lenc = LabelEncoder()

df['month'] = lenc.fit_transform(df['month'])
df['day'] = lenc.fit_transform(df['day'])


#//// Normalizing the dataset

def normalize(i):
    x = (i - i.min())/(i.max() - i.min())
    return(x)

ip = normalize(df.iloc[:, :10])
op = df['area']

###############################################################################

#//// Splitting the data
from sklearn.model_selection import train_test_split

train_ip, test_ip, train_op, test_op = train_test_split(ip, op, test_size = 0.2)

def forest():
    model = Sequential()
    model.add(Dense(64, activation = 'relu', input_shape = (ip.shape[1],)))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1))
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    return model
    

k = 4
samples = len(train_ip) // k    
num_epochs = 100
all_score = []    

for i in range(k):
    print('processing fold : ', i)
    train_data = train_ip[i*samples : (i+1)*samples]
    train_targ = train_op[i*samples : (i+1)*samples]
    partial_train_data = np.concatenate([train_ip[:i*samples], train_ip[(i+1)*samples:]], axis = 0)
    partial_train_targ = np.concatenate([train_op[:i*samples], train_op[(i+1)*samples:]], axis = 0)
    model = forest()
    model.fit(partial_train_data, partial_train_targ, epochs = num_epochs, batch_size = 2, verbose = 0)
    val_mse, val_mae = model.evaluate(train_data, train_targ, verbose = 0)
    all_score.append(val_mae)

all_score
np.mean(all_score)


num_epochs = 100
all_mae_histories = []

for i in range(k):
    print('processing fold : ', i)
    train_data = train_ip[i*samples : (i+1)*samples]
    train_targ = train_op[i*samples : (i+1)*samples]
    partial_train_data = np.concatenate([train_ip[:i*samples], train_ip[(i+1)*samples:]], axis = 0)
    partial_train_targ = np.concatenate([train_op[:i*samples], train_op[(i+1)*samples:]], axis = 0)
    model = forest()
    history = model.fit(partial_train_data, partial_train_targ, epochs = num_epochs, validation_data = (train_data, train_targ), batch_size = 2, verbose = 0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
    

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

    
#Training the final model
model = forest()
model.fit(train_data, train_targ, epochs=2, batch_size=4, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_ip, test_op)
test_mse_score, test_mae_score

###############################################################################