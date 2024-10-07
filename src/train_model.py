import numpy as np
import math
class Regression:
    def __init__(self,dimensions,iterations=100000, regularizer = 0.00001,learning_rate = 0.0001):
        self.regularizer = regularizer 
        self.lr = learning_rate
        self.weights = np.zeros(dimensions)
        self.weights.reshape(1,dimensions)
        self.dimensions = dimensions
        self.bias = 0
        self.iterations = iterations
    
    def predict(self,X):
        assert X.shape[1]==len(self.weights) ,"Please input a numpy array that is equal in length to the number of weights"
        return np.dot(X,self.weights) + self.bias
    
    def regularizer_function(self): #penalty added to cost function to prevent overfitting
        return self.regularizer * np.sum(np.square(self.weights))
     
    def gradient_descent(self,X,y): #gradient descent algorithm
        for i in range(self.iterations):

            predictions = np.dot(X,self.weights) + self.bias 
            errors = y - predictions 

            dw = ((1/X.shape[0]) * np.dot(X.T,errors)) + (self.regularizer*2*self.weights) #derivative of cost function w.r.t to weights
            db = (1/X.shape[0])* np.sum(errors) #derivative of cost function w.r.t bias

            self.weights = self.weights + self.lr*dw
            self.bias = self.bias + self.lr*db
    
    def mse(self,X,target): #function to calculcate mse
        predictions = self.predict(X)
        return np.mean((target-predictions)**2)
    
    def root_mse(self,X,target): #function to calculate root mse
        return math.sqrt(self.mse(X,target))
    
    def r2(self,X,target): #function to calculate r2 score
        predictions = self.predict(X)
        residual_sum = np.sum((target-predictions)**2)
        mean = np.mean(target)
        total_sum = np.sum((mean-predictions)**2)
        return 1 - (residual_sum/total_sum)
    



