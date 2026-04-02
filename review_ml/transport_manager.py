
#importing required libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

#connecting with drive to load data
from google.colab import drive
drive.mount("/content/drive")

#loadind data
df=pd.read_csv("/content/drive/MyDrive/aws_inventory_logistics_raw.csv")
df.head()

df.info()

#handling nulls
df=df.dropna()
df.isnull().sum()

x=df.drop("transport_cost",axis=1)
x=pd.get_dummies(x,drop_first=True)

y=df["transport_cost"]

#scalar
scaler=StandardScaler()
X=scaler.fit_transform(x)

#splitting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#training model
model=LinearRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

#checking accuracy of the model
#r2score
r2score=r2_score(y_test,y_pred)
print("r2_score:",r2score)

#mean squared error
mse_error=mean_squared_error(y_test,y_pred)
print("mean_squred_error:",mse_error)

#mean absolute error
mae_error=mean_absolute_error(y_test,y_pred)
print("mean_absolute_error",mae_error)

#------------ Random forest----------------------



from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
from sklearn.compose import ColumnTransformer


#calculating required features
df["cost_cat"]=df["transport_cost"].apply(
    lambda x:"High" if x>df['transport_cost'].mean() else 'Low'
)
x=df.drop(["transport_cost","cost_cat"],axis=1)
x=pd.get_dummies(x,drop_first=True)

y=df["cost_cat"]

#selecting object datatypes
cat_features=x.select_dtypes(include=['object']).columns
num_features=x.select_dtypes(include=['int64','float64']).columns

preprocessor=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),num_features),
        ('cat',OneHotEncoder(handle_unknown='ignore'),cat_features)
    ]
)

#creating pipeline
pipeline=Pipeline(steps=[('preprocessor',preprocessor),
                          ('model',RandomForestClassifier(n_estimators=100,random_state=42))
])
#model.fit(x_train,y_train)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train model
pipeline.fit(x_train,y_train)
#predict
y_pred=pipeline.predict(x_test)

#checking accuracy
print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

