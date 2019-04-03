
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture 
import matplotlib.pyplot as plt
from pandas import DataFrame 

df = pd.read_csv('health.csv',nrows=500,encoding = "ISO-8859-1")

"""
def WorkTypeMapping(df):
    df['Work Type int']=''
    for i in range(len(df['Work Type'])):
        if df['Work Type'].iloc[i]=='Private':
            df['Work Type int'].iloc[i]=1
        elif df['Work Type'].iloc[i]=='children':
            df['Work Type int'].iloc[i]=2
        elif df['Work Type'].iloc[i]=='Govt_job':
            df['Work Type int'].iloc[i]=3
        elif df['Work Type'].iloc[i]=='Never_worked':
            df['Work Type int'].iloc[i]=4
        else:
            df['Work Type int'].iloc[i]=5
# turn it into a dataframe
def GenderMapping(df):
    df['Gender int']=''
    for i in range(len(df['Gender'])):
        if df['Gender'].iloc[i]=='Male':
            df['Gender int'].iloc[i]=10
        else:
            df['Gender int'].iloc[i]=20

# GenderMapping(df)
# WorkTypeMapping(df)
"""
def clustering(df):

    d = df[['Age','BMI']]
    # print(d)

    p1=plt.figure(1)
    plt.scatter(d['Age'],d['BMI'])

    gmm = GaussianMixture(n_components = 3) 
  
# Fit the GMM model for the dataset  
# which expresses the dataset as a  
# mixture of 3 Gaussian Distribution 
    gmm.fit(d) 
    labels = gmm.predict(d) 
    d['labels']= labels
    d0 = d[d['labels']== 0] 
    d1 = d[d['labels']== 1] 
    d2 = d[d['labels']== 2]
    df1 = pd.merge(df, d0, on=['BMI','Age'], how='inner')
    df2 = pd.merge(df, d1, on=['BMI','Age'], how='inner')
    df3 = pd.merge(df, d2, on=['BMI','Age'], how='inner')
    print(df1)
    # print(df2)
    # print(df3)
    df1=df1.drop(columns='labels')
    # print(df1)
    df2=df2.drop(columns='labels')
    df3=df3.drop(columns='labels')
    p2=plt.figure(2)
    plt.scatter(d0['Age'],d0['BMI'], c ='r') 
    plt.scatter(d1['Age'],d1['BMI'], c ='yellow') 
    plt.scatter(d2['Age'],d2['BMI'], c ='g') 

    plt.show()
# print the converged log-likelihood value 
    print(gmm.lower_bound_) 
  
# print the number of iterations needed 
# for the log-likelihood value to converge 
    print(gmm.n_iter_)
    return [df1,df2,df3]

def slicing(df):
    df=combine(df,['Age','BMI','Work Type'])
    df=permute(df,'Age,BMI,Work Type')
    df=split(df,'Age,BMI,Work Type')
    df=combine(df,['Hypertension','Heart Disease','Smoking Status','Glucose'])
    df=permute(df,'Hypertension,Heart Disease,Smoking Status,Glucose')
    df=split(df,'Hypertension,Heart Disease,Smoking Status,Glucose')
    print(df)
    return df

def supress(df):
    df.at[:,'Name']='*'
    print(df,'\n')
    return df

def generalize(df):
    df['Age']=df['Age'].astype(int)
    for i in range(len(df['Age'])):
        if df['Age'].iloc[i]>0 and df['Age'].iloc[i]<=15:
            df['Age'].iloc[i]='<15'
        elif df['Age'].iloc[i]>15 and df['Age'].iloc[i]<=25:
            df['Age'].iloc[i]='15-25'
        elif df['Age'].iloc[i]>25 and df['Age'].iloc[i]<=40:
            df['Age'].iloc[i]='25-40'
        elif df['Age'].iloc[i]>40 and df['Age'].iloc[i]<=60:
            df['Age'].iloc[i]='40-60'
        elif df['Age'].iloc[i]>60 and df['Age'].iloc[i]<=75:
            df['Age'].iloc[i]='60-75'
        elif df['Age'].iloc[i]>=70:
            df['Age'].iloc[i]='> 75'
    print(df,'\n')
    return df

def permute(df,colname):
    df[colname] = np.random.permutation(df[colname])
    # print(df)
    return df



def combine(df,cols):
    colname=','.join(cols)
    df[colname]=''
    for i in range(len(cols)):
        if i==0:
            df[colname]=df[cols[i]].astype(str)
        else:
            df[colname]=df[colname].astype(str)+','+df[cols[i]].astype(str)
    # print(df)
    return df
     


def split(df,colname):
    cols=colname.split(",")
    new = df[colname].str.split(",", expand = True)
    for i in range(len(cols)):
        df[cols[i]]=new[i]
    df=df.drop(columns=colname)
    # print(df)
    return df

def anatomize(df):
    df1=df[['Id', 'Age','BMI','Work Type']]
    df2=df[['Id', 'Hypertension','Heart Disease','Smoking Status','Glucose']]
    print(df1,'\n')
    print(df2,'\n')
    return df

"""
print('Before slicing\n',df)
df=slicing(df)
print('After slicing\n',df)
"""

dfs=clustering(df)
for i in range(len(dfs)):
    dfs[i]=slicing(dfs[i])
    dfs[i]=supress(dfs[i])
    dfs[i]=generalize(dfs[i])
    dfs[i]=anatomize(dfs[i])
    dfs[i].to_csv("health"+str(i)+".csv")

dfnew=pd.concat([df for df in dfs])
# dfnew.drop_duplicates(subset ="Id", 
#                      keep = False, inplace = True) 
# print(dfnew)
dfnew=dfnew.drop(columns='Unnamed: 0')
dfnew.to_csv("healthnew.csv")



# print('After slicing\n',df,'\n')
# supress(df)
# generalize(df)
# anatomize(df)

