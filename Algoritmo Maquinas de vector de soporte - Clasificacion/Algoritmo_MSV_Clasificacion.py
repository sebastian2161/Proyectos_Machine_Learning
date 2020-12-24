import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

df = pd.read_csv('train_v2.csv')
#print(df)

#df.columns
#print(df.columns)
#print(df.shape)


#Posiciones de de las filas y columnas de la matriz.

#print(df.iloc[5]) # Fila en la posición 5, es decir, la 6ª fila.
#print(df.iloc[:2]) # Filas en el rango [0,2), la fila en posición 2 no es incluida.
#print(df.iloc[0,0]) # Celda en la posición (0,0).
#print(df.iloc[[0,10,12],3:6])


#print(df.loc[0]) # Fila con índice 0.
#print(df.loc[0,'Fare']) # Fila con índice 0 y columna Fare.

#print(df.loc[:3, 'Sex':'Fare']) 
# Filas con índices en el rango [0,3] y columnas
#entre Sex y Fare. En este caso ambos extremos se incluyen, tanto en filas
#como en columnas.

#print(df.loc[:3, ['Sex','Fare','Embarked']]) 
# Filas con índices en el rango
#[0,3] y columnas con nombre Sex, Fare y Embarked.

#print(df.dtypes) #Tipos de variables cabeceras

#print(df.describe())  
#print(df.describe(include='all'))

#df = df.drop(columns=['PassengerId', 'Name', 'Ticket','Cabin'])
#print(df.describe(include='all'))

#df = df.dropna() #Elimina algun valor vacio de una columna
#print(df)

#df['Sex'] = df['Sex'].astype('category').cat.codes
#df['Embarked'] = df['Embarked'].astype('category').cat.codes

#df.to_csv('titanic_ml.csv', index=False)

def split_label(df, test_size, label, label1):

     train, test = train_test_split(df,test_size=test_size)
     #features = df.columns.drop(label)
     features = df.columns.drop([label1, label])
     #features = df.drop(columns=['Name','Survived'])
     #print(features)
     train_X = train[features]
     train_y = train[label]
     test_X = test[features]
     test_y = test[label]
     #print(train_X)
     #print(train_y)
     #print(test_X)
     #print(test_y)
     return train_X, train_y, test_X, test_y



def one_hot():
     X = train_X.iloc[:,:].values

     print(train_X.iloc[:,2])
     #print(train_X.iloc[:,9])

     ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2])], remainder='passthrough')
     X = np.array(ct.fit_transform(X))
     #print(X)

     #ct = ColumnTransformer(transformers=[('encoder1',OneHotEncoder(),[9])], remainder='passthrough')
     #X = np.array(ct.fit_transform(X))

     #print(X)


def one_hot_v1():
     cols = ['PassengerId','Pclass','Fare', 'Embarked', 'Sex', 'Age']
     X = train_X[cols]

     ohe = OneHotEncoder()
     imp = SimpleImputer()

     ct = make_column_transformer(
          (ohe, ['Embarked', 'Sex']),  # apply OneHotEncoder to Embarked and Sex
          (imp, ['Age']),              # apply SimpleImputer to Age
          remainder='passthrough')     # include remaining column (Fare) in the output
     X = ct.fit_transform(X)
     
     min_max_scaler = MinMaxScaler()
     X1 = min_max_scaler.fit_transform(X)

     from sklearn.svm import SVC
     clf = SVC()
     clf.fit(X1, train_y)

     cols1 = ['PassengerId','Pclass','Fare', 'Embarked', 'Sex', 'Age']
     X_t = test_X[cols1]

     ohe1 = OneHotEncoder()
     imp1 = SimpleImputer()

     ct = make_column_transformer(
          (ohe1, ['Embarked', 'Sex']),  # apply OneHotEncoder to Embarked and Sex
          (imp1, ['Age']),              # apply SimpleImputer to Age
          remainder='passthrough')     # include remaining column (Fare) in the output
     X_t = ct.fit_transform(X_t)

     min_max_scaler1 = MinMaxScaler()
     X_t1 = min_max_scaler1.fit_transform(X_t)

     print(clf.predict(X_t1))
     print(X_t1)
     print(clf.score(X_t1, test_y))
     #print(test_y)



train_X, train_y, test_X, test_y = split_label(df, 0.2, 'Survived', 'Name')
one_hot_v1()





