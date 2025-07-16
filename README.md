# Overview
This project aims to predict stock prices using Long Short Term Memory (LSTM) models. The models use various stock price indicators as training data, some of these indicators include:

RSI,AROOND_14,AROONU_14,AROONOSC_14,MACD_12_26_9,MACDh_12_26_9,MACDs_12_26_9,ADX_14,DMP_14,DMN_14

This project also implements incremental updating of the model, so whenever new stock data comes in you can easily update your model with this new data. 

There is also a ipynb which provides a simple template for creating new recurrent neural network models, so you can edit parameters such as the optimizer, loss function, epochs, train/test split, model layers, dropout, etc.

This is a CLI interactive program. When you run the main file you will be prompted with questions, asking if you would like to perform a predicition or update a model. It will also tell you if there is new data available. Then it will give you a prediction and graph its past predicitions, and future, against the actual stock price.

## Files
create_new_model.ipynb - skeleton for creating new models

make_model_pipeline.py - saves the model in .keras format, as well as a json with the model's details which continuously gets updated, also saves predicitions/new data to csv

my_models - Directory with model folders which contain files saved from above file

createDS.py - Creates Dataset, uses moving window so LSTM accepts data. Can add other indicators here

test_pipeline.py - Performs the predictions, comparisons, as well as creates the plots

train_or_predict.py - Main file to run the program. Don't need to touch other files

## Technology
tensorflow, keras, LSTM, yfinance, pandas, pandas_ta, numpy, matplotlib, sklearn

## Future Work
1. Make the updating of the model more automated so dont have to run the program every day
2. Email notifications of predictions


# How to Run

1. Clone Repo down
2. Create venv using below commands (python3.10.18)
```
python3.10 -m venv venvname
source venvname/bin/activate
pip install cython==0.29.32 
pip install --no-build-isolation jnius==1.1.0
pip install -r requirements.txt
```
3. Run train_or_predict.py
4. Interact through terminal (all y or n answers)

Note: The LSTM.keras models may get corrupted when cloned down, to create a new model just run trhough create_new_model.ipynb, very simple.

## Output

Prompted to choose a model, then you can choose to see what data would be going into it before deciding to update it or not
<img width="991" height="398" alt="Screenshot 2025-07-15 at 10 09 52 PM" src="https://github.com/user-attachments/assets/c09e78ee-f408-4101-93b8-298d27eb1e17" />

After training the model with the updated data you can either wait until after seeing its predicition to save it or save it now. Then you can see the models predicition for tomorrow, not very accurate lol, this was a quick model with only like 8 epochs.
<img width="878" height="114" alt="Screenshot 2025-07-15 at 10 15 30 PM" src="https://github.com/user-attachments/assets/97223b83-c7a3-49c8-98af-a8b639d477a2" />

After saving model and predicition you can choose to see a plot
<img width="346" height="54" alt="Screenshot 2025-07-15 at 10 19 46 PM" src="https://github.com/user-attachments/assets/f4be5a01-942e-451f-8aec-50451ac37aee" />


<img width="645" height="481" alt="Screenshot 2025-07-15 at 10 20 31 PM" src="https://github.com/user-attachments/assets/71baf6ed-4ae5-41b2-839a-2c9e8fbbb1ec" />

As you can see our model has predicted that this stocks price goes down tomorrow.







