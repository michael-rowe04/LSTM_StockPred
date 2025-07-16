import json
import tensorflow as tf
import keras
import os

class make_model():
    def __init__(self,model_name,model,optimizer,loss,epochs,batch_size,stock,period,train_data_name,train_data,start_data,end_data,num_train_points):
        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.stock = stock
        self.period = period
        self.train_data_name = train_data_name
        self.train_data = train_data
        self.start_data = start_data
        self.end_data = end_data
        self.num_train_points = num_train_points


    def create_record(self):
        train_before_save = {"optimizer":self.optimizer,"loss":self.loss,"epochs":self.epochs,"batch_size":self.batch_size}
        data_before_save = {"stock":self.stock,"period":self.period,"train_data_name":self.train_data_name,"train_data":self.train_data,"start_data":self.start_data,"end_data":self.end_data,"num_train_points":self.num_train_points}
        data_after_save = {"date_last_trained":self.end_data,"updated_data_len":0}
        big_dict = {"train_before_save":train_before_save,"data_before_save":data_before_save,"data_after_save":data_after_save}
        with open("my_models/"+ self.model_name + "/" + self.model_name + ".json","w") as outfile:
            json.dump(big_dict,outfile,indent = 6)

    def save_model(self):
        print("Saving model: " + self.model_name)
        os.makedirs("my_models/"+self.model_name+"/", exist_ok=True)
        self.model.save("my_models/"+ self.model_name + "/" + self.model_name + ".keras")

    