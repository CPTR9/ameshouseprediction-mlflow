***House Price Prediction***
To Run :
Terminal 1> mlflow ui --host 0.0.0.0 --port 5000 

Terminal 2>python main.py

Get Model path from MLFlow>Model>Artifacts

update  predict.py>logged_model(line4) with model path

update processed_file.csv file

IMP : Use only pre processed data for predictions

python predict.py 

