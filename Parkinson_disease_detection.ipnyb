import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

#loading the data from csv file to pandas dataframe
parkinsons_data = pd.read_csv('/content/parkinsons.csv')

#printing first 5 data from file
parkinsons_data.head()

#number of rows and colums in a dataframe
parkinsons_data.shape

etting more information
parkinsons_data.info()

#checking for missing values
parkinsons_data.isnull().sum()

#getting some statistical datas about data
parkinsons_data.describe()

#distribution of target variable

parkinsons_data['status'].value_counts()

#grouping the data based on target variable
parkinsons_data.groupby('status').mean()

#seperating the features and target
X=parkinsons_data.drop(columns=['name','status'],axis=1)
Y=parkinsons_data['status']
print(X)

print(Y)

#splitting the training data & test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape, X_train.shape, X_test.shape)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

print(X_train)

#model training
#support vector machine model
model = svm.SVC(kernel='linear')

#training the model
model.fit(X_train,Y_train)

#model evaluation
#accuracy score
X_train_prediction=model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_prediction)

print('accuracy score of training data', training_data_accuracy)


#accuracy score
X_test_prediction=model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test,X_test_prediction)

print('accuracy score of training data',test_data_accuracy)

 #building a predictive system
input_data=(108.80700,134.65600,102.87400,0.00761,0.00007,0.00349,0.00486,0.01046,0.02719,0.25500,0.01483,0.01609,0.02067,0.04450,0.01036,21.02800,0.536009,0.819032,-4.649573,0.205558,1.986899,0.316700)

#changing input data into numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizer the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print("the person doesnot have parkinsons disease")
else:
  print("the person have parkinsons disease")
