from data_preprocessing import *
from matplotlib import pyplot as plt
from train_model import Regression
import pickle
import csv

def createMetricsFile(model,data,targets): #function to create metrics file
    
    metrics = open("./results/metrics.txt",'w')
    metrics.write("Regression Metrics: \n")

    metrics.write(f"Mean Squared Error (MSE): {round(model.mse(data,targets),2)} \n")
  

    metrics.write(f"Root Mean Squared Error (RMSE): {round(model.root_mse(data,targets),2)} \n")


    metrics.write("R-Squared (R\u00b2) Score: " + f" {round(model.r2(data,targets),2)}")

    metrics.close()

def createPredictionsCsv(model,X): #function to create csv file
    predictions = model.predict(X)
    with open('./results/predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for prediction in predictions:
            writer.writerow([prediction])


DATA =getData()
training_data,training_targets,test_data,test_target = test_training_split(DATA,80)
 
with open('./models/regression_model_final.pkl','rb') as file:
    regressor = pickle.load(file)

regressor.gradient_descent(training_data,training_targets)

createMetricsFile(regressor,test_data,test_target)
createPredictionsCsv(regressor,test_data)