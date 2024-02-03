import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import KMeansSMOTE
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import silhouette_score
df=pd.read_csv("sensor_data.csv")
le=LabelEncoder()
df["Boiler Name"]=le.fit_transform(df["Boiler Name"])
#df.info()
#df['Timestamp']
import datetime
df['Timestamp']=pd.to_datetime(df['Timestamp'],format='%d-%m-%Y %H:%M')
df['Timestamp']=df['Timestamp'].map(datetime.datetime.toordinal)
df['Timestamp']=df['Timestamp'].astype(int)
#df['Timestamp']
kmeans1=KMeansSMOTE()
x,y=kmeans1.fit_resample(df,df['Anomaly'])
print(y.value_counts())
plt.boxplot(df['Temperature'])
df=x
x=x.iloc[:,[1,2,3]].values
wcss=[]
for i in range(1,15):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
kmeans=KMeans(n_clusters=4,random_state=0)
ymeans=kmeans.fit_predict(x[:,[0,1,2]])
df['clusters']=ymeans
silhouette_score(x,ymeans)
but=st.sidebar.button("Depending upon temperature and anomaly,clusters will be assigned")
if but:
    fig,ax=plt.subplots(figsize=(10,10))
    ax=sns.scatterplot(x="Temperature",y="Anomaly",hue="clusters",data=df,palette="rainbow")
    st.pyplot(fig)
but2=st.sidebar.button("Depending upon temperature and boiler name,clusters will be assigned")
if but2:
    fig,ax=plt.subplots(figsize=(10,10))
    ax=sns.scatterplot(x="Temperature",y="Boiler Name",hue="clusters",data=df,palette="rainbow")
    st.pyplot(fig)
but3=st.sidebar.button("Predict to which cluster the given data belongs to")
ymeans1=""
col1,col2=st.columns(2)
if but3:
    t=col1.text_input("Enter Temperature",value=20.18042818)
    b=col1.text_input("Enter Boiler Name",value="A")
    a=col1.text_input("Enter Anomaly",value=0)
    t1=float(t)
    a1=int(a)
    if b=="A":
        b1=0
    elif b=="B":
        b1=1
    elif b=="C":
        b1=2
    elif b=="D":
        b1=3
    else:
        col1.error("Enter valid boiler name")
    ymeans1=kmeans.predict([[b1,t1,a1]])
    #but4=col1.button("Predict")
    
    
            
        
    col1.text_area("Predicted Cluster:",ymeans1[0])
        # if ymeans1[0]==0:
        #     st.write("Cluster A")
        # elif ymeans1[0]==1:
        #     st.write("Cluster B")
        # elif ymeans1[0]==2:
        #     st.write("Cluster C")
        # elif ymeans1[0]==3:
        #     st.write("Cluster D")
