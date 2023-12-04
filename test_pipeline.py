import pandas as pd
import matplotlib.pyplot as plt
class tester():
    def __init__(self, y_actuals,predictions,model,X_test):
        self.model = model
        self.y_actuals = y_actuals
        self.predictions = predictions
        self.X_test = X_test

    def compare(self):
        #Create a df of my actuals and preds so I can drop that last NA
        my_df = pd.DataFrame()
        my_df['Actual'] = self.y_actuals
        my_df['Preds'] = self.predictions
        my_df = my_df.dropna()

        right_direction = 0
        index = 0
        my_dict = {}
        for i,j in zip(my_df['Preds'],my_df['Actual']):
            bool_direction = False
            i = float(i) #preds returns all preds in a numpy array
            #i=pred, j= actual
            if((i > 0 and j > 0) or (i < 0 and j < 0)):
                right_direction += 1
                bool_direction = True
            my_dict[index] = [i,j,abs(i-j),bool_direction]
            index += 1
        #Put everything calculated above in a data frame so I can see visually how everything did
        new_df = pd.DataFrame.from_dict(my_dict,orient = 'index', columns = ["Prediction","Actual","Error","Right Direction"])
        print(new_df)
        accuracy = 100*(right_direction/len(my_df)) #just checks how many time it got the accuracy right
        print("This model got the direction right" , right_direction, "times out of", len(my_df),"for a whopping accuracy of",accuracy,"%")
    
    def plot(self):
        #this still plots the final prediction that is matched up with the nan, that way you can see which way you think it will go, nan doesnt show up
        x_cord = [i for i in range(len(self.predictions))]
        x_axis = [0 for i in range(len(self.predictions))]
        plt.plot(x_cord,self.predictions,label = "pred")
        plt.plot(x_cord,self.y_actuals, label = "real")
        plt.plot(x_cord,x_axis)
        plt.legend()
        plt.show()


    def incremental_updating(self):
        predictions = []
        y_test_list = []
        index = 0
        for input_index in range(1,len(self.X_test)):
            #Have index by index:input_index so it can be a dataframe/series. its just getting one line of data though, i.e. [0:1]
            
            #this makes the prediction firs then trains it
            #y_pred = self.model.predict(self.X_test.iloc[index:input_index])
            #print(y_pred)
            #self.model.fit(self.X_test.iloc[index:input_index],self.y_actuals.iloc[index:input_index],batch_size = 1, epochs = 1)
            
            self.model.fit(self.X_test.iloc[index:input_index],self.y_actuals.iloc[index:input_index],batch_size = 1, epochs = 1)
            y_pred = self.model.predict(self.X_test.iloc[index+1:input_index+1])
            print(y_pred)
            
            #predictions[index] = [float(y_pred),y_test.iloc[index+1:input_index+1].iloc[0]]
            predictions.append(float(y_pred))
            y_test_list.append(self.y_actuals.iloc[index+1:input_index+1].iloc[0])
            index +=1
        return predictions,y_test_list