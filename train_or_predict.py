import os
import json
from createDS import dataset
import tensorflow as tf
import keras
from keras import models
import pandas as pd
import matplotlib.pyplot as plt
#models = os.listdir('/Users/michael.rowe/Personal/IntrinsicValueProject/my_models')
#venv is mtenv
class update_or_predict():

    def __init__(self):
        self.models = os.listdir('my_models')

    def printchoices(self,models):
        for model in models:
            print(models.index(model), ": " , model)
        choice = input("Please choose a model: ")
        the_model = models[int(choice)]
        #the_model = "my_models/" + the_model + "/" + the_model
        return the_model
    
    def create_csv(self,the_model,todays_data,prediction):
        todays_data_and_pred = pd.DataFrame(todays_data)
        todays_data_and_pred['Prediction'] = prediction
        try:
            pd.read_csv("my_models/" + the_model + "/" + the_model + ".csv")
            todays_data_and_pred.to_csv("my_models/" + the_model + "/" + the_model + ".csv",mode = 'a',header = False)
        except:
            todays_data_and_pred.to_csv("my_models/" + the_model + "/" + the_model + ".csv",mode = 'a',header = True)
    
    def make_plot(self,the_model,y_train): #need to make sure >1 points of data and need to have the real data for all my predictions,
        pred_csv = pd.read_csv("my_models/" + the_model + "/" + the_model + ".csv")
        pred = list(pred_csv["Prediction"]) #have to remember in the csv the X data is predicting data for the next day, which we dont always have access to
        actuals = []
        for date in pred_csv['Date']:
            actuals.append(y_train[date])
        x_cord = [i for i in range(len(pred))] #I think we can only run this when we have > 1 points of data
        x_axis = [0 for i in range(len(pred))]
        plt.plot(x_cord,pred,"-o",label = "pred")
        plt.plot(x_cord,actuals,"-o", label = "real") #something wrong with this, if i were to plot when i have a prediction but not the real data yet there would be an issue
        plt.plot(x_cord,x_axis)
        plt.legend()
        plt.show()


    def updated_train_data(self,the_model):
        training_up_to_date = False #Probably a better way to do this but otherwise I am getting an indexing error so am going to do a try catch
        model_dict = json.load(open("my_models/" + the_model + "/" + the_model + ".json"))
        X_train, X_test, y_train, y_test, todays_data = dataset(model_dict["data_before_save"]["stock"],model_dict["data_before_save"]["period"],model_dict["data_before_save"]["train_data_name"]).split(1)
        actuals = y_train #all the actual values
        X_train = X_train.loc[model_dict['data_after_save']['date_last_trained']:].iloc[1:-1] #there will be no data if training data is last from today
        y_train = y_train.loc[model_dict['data_after_save']['date_last_trained']:].iloc[1:-1] #need to drop the last data point because that is todays data and it would be trained on a nan for the pred since we dont have tmrws data yet
        #last_day_trained_on = str(X_train.iloc[-1]).split("Name:")[-1].split(", ")[0]
        last_day_trained_on = model_dict['data_after_save']['date_last_trained'] #before model was just picking up the last day it is going to be trained on
        try:
            today = str(X_train.loc[model_dict['data_after_save']['date_last_trained']:].iloc[-1]).split("Name:")[-1].split(", ")[0] #today isnt technically today, its the last that the model can be trained on
        except:
            today = last_day_trained_on #because if training is up to date then X_train will be empty, so cant pull today
        #except:
        #today = str(X_train.loc[model_dict['data_after_save']['date_last_trained']:].iloc[-1]).split("Name:")[-1].split(", ")[0] #this just gives today, could do it with less code but dont feel like initializing stuff
        #today = model_dict['data_after_save']['date_last_trained'] #above line was reading from X_train, but if X_train is up to date this index will be out of date, so i am just pulling from the dict
        #but then today is today, it would be the day last trained so we cant do this we need it to be today.
        if last_day_trained_on == today:
            training_up_to_date = True
        return X_train,y_train,todays_data,last_day_trained_on,training_up_to_date,today,actuals

    def load_model(self,the_model):
        loaded_model = tf.keras.models.load_model("my_models/" + the_model + "/" + the_model + ".keras")
        return loaded_model

    def train_model(self,loaded_model,X_train,y_train):
        #X_train, y_train = updated_train_data(the_model)
        loaded_model.fit(X_train,y_train,batch_size = 1, epochs = 1)
        return loaded_model

    def do_predict(self,trained_or_loaded_model,todays_data):
        pred = trained_or_loaded_model.predict(todays_data)
        return pred

    def update_dict(self,the_model,last_day_trained_on,X_train):
        the_dict = json.load(open("my_models/" + the_model + "/" + the_model + ".json"))
        the_dict['data_after_save']['date_last_trained'] = last_day_trained_on
        the_dict['data_after_save']["updated_data_len"] = the_dict['data_after_save']["updated_data_len"] + len(X_train)
        with open("my_models/" + the_model + "/" + the_model + ".json","w") as outfile:
            json.dump(the_dict,outfile,indent = 6)

    def train_or_predict(self,loaded_model,X_train,y_train,todays_data,the_model,last_day_trained_on,training_up_to_date,today,actuals):
        print(the_model + "was last trained on data from: " + last_day_trained_on)
        ans10 = input("Would you like to see what will be going into the model?: ")
        if ans10 == 'y': 
            print("X_train: " , X_train)
            print(type(X_train))
            print(type(y_train))
            print("y_train: " , y_train)
            print("Todays data: " , todays_data)
            print(type(todays_data))
            print("last_day_trained_on: " , last_day_trained_on)
            print("training_up_to_date: " , training_up_to_date)
            print("Today: " , today)

        ans = input("Would you like to train the model: ")
        if ans == 'y' and training_up_to_date == False:
            self.train_model(loaded_model,X_train,y_train)
            print("Model has been trained.")
            ans4 = input("Would you like to save your newly trained model now or after making a prediction, your old model will be replaced: ")
            if ans4 == 'y':
                loaded_model.save("my_models/" + the_model + "/" + the_model + ".keras")
                self.update_dict(the_model,str(X_train.iloc[-1]).split("Name:")[-1].split(", ")[0],X_train) # Updating after save instead of after training
                print("Save done.")
            ans2 = input("Would you like to predict the stock price for tomorrow: ")
            if ans2 == 'y':
                pred = self.do_predict(loaded_model,todays_data)
                print(pred)
                if ans4 == 'n':
                    ans5 = input("Would you like to save your model now: ")
                    if ans5 == 'y':
                        loaded_model.save("my_models/" + the_model + "/" + the_model + ".keras")
                        self.update_dict(the_model, str(X_train.iloc[-1]).split("Name:")[-1].split(", ")[0], X_train)
                    elif ans5 == 'n':
                        print("okay")
                ans8 = input("Would you like to save your prediction: ")
                if ans8 == 'y':
                    self.create_csv(the_model,todays_data,pred)
            ans9 = input("Would you like to see a plot: ")
            if ans9 == 'y':
                self.make_plot(the_model,actuals) #can only run with 2+ data in csv
            if ans2 == 'n':
                #logic for if you want to save new trained model, also need logic for updating dict
                print("okay")
        elif ans == 'n' or training_up_to_date == True:
            if training_up_to_date == True:
                print("Training all up to date.")
            ans3 = input("Would you like to make a prediction: ")
            if ans3 == 'y':
                pred = self.do_predict(loaded_model, todays_data)
                print(pred)
                ans6 = input("Would you like to save the prediction: ")
                if ans6 == 'y':
                    self.create_csv(the_model,todays_data,pred)
            ans7 = input("Would you like to see a plot: ")
            if ans7 == 'y':
                self.make_plot(the_model,actuals) #need 2+ data in csv

    def run(self):
        the_model = self.printchoices(self.models)
        X_train,y_train,todays_data,last_day_trained_on,training_up_to_date,today,actuals = self.updated_train_data(the_model)
        loaded_model = self.load_model(the_model)
        self.train_or_predict(loaded_model,X_train,y_train,todays_data,the_model,last_day_trained_on,training_up_to_date,today,actuals)


""" the_model = printchoices(models)
X_train,y_train,todays_data,last_day_trained_on,training_up_to_date = updated_train_data(the_model)
loaded_model = load_model(the_model)
train_or_predict(loaded_model,X_train,y_train,todays_data,the_model,last_day_trained_on,training_up_to_date)
#print(last_day_trained_on) """

doit = update_or_predict()
doit.run()
