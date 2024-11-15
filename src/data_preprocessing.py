import pandas as pd
from category_encoders import TargetEncoder

def getData():
    data = pd.read_csv('./data/fuel_train.csv')
    data.drop("Year",axis = 1,inplace = True) #dropping redundant columns
    data.drop("MODEL",axis = 1,inplace = True)
    data = pd.get_dummies(data,columns = ['FUEL','TRANSMISSION']) #performing one-hot encoding
    data = data.replace({True: 1, False: 0}) 

    
    cols = ['MAKE','VEHICLE CLASS']
    for col in cols: #performing target encoding on categorical variable with multiple categories
        te = TargetEncoder()
        te.fit(X=data[col],y = data['FUEL CONSUMPTION'])
        values = te.transform(data[col])
        data = pd.concat([data,values],axis = 1)
    

    data.drop(data.columns[0],axis = 1,inplace=True) #dropping categorical columns
    data.drop(data.columns[0],axis = 1,inplace=True)

    data['ENGINE SIZE'] = (data['ENGINE SIZE'] - data['ENGINE SIZE'].mean()) / data['ENGINE SIZE'].std() #normalizing the data
    data['CYLINDERS'] = (data['CYLINDERS'] - data['CYLINDERS'].mean()) / data['CYLINDERS'].std()
    data[data.columns[3]] = (data[data.columns[3]] - data[data.columns[3]].mean()) / data[data.columns[3]].std()

    column_order = data.columns.tolist() #rearranging so target variable is the last column
    column_order[-1],column_order[2] = column_order[2],column_order[-1]
    data = data.reindex(columns = column_order)


    return shuffle_data(data)

def shuffle_data(data): #shuffles the date randomly
    return data.sample(frac=1).reset_index(drop=True)

def test_training_split(data,split_percentage): #returns training and test data
    index = int((split_percentage/100)*data.shape[0])
    h,t= data[:index],data[index:]  
    return (h.iloc[:,0:-1].to_numpy(),h.iloc[:,-1].to_numpy(),
            t.iloc[:,0:-1].to_numpy(),t.iloc[:,-1].to_numpy())
