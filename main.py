import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import psutil
import time

mlflow.set_tracking_uri("http://localhost:5000")
scaler = StandardScaler()

def preprocess_data(df):
    print("Number of data samples(Rows):", df.shape[0])
    print("Number of features(Coloumns):", df.shape[1])

    for column in df.columns:
        nullrows = df[column].isnull().sum()
        if(nullrows > 400):
            df.drop(column , axis = 1 , inplace = True)
            print(column,"Dropped")
            print('********************************************************')
        
    print("Number of features(Coloumns) after dropping columns with large number of empty values:", df.shape[1])
    print('Dropping Order and PID columns')
    df.drop(['Order','PID'], axis=1 , inplace = True )

    for column in df.columns:
        nullrows = df[column].isnull().sum()
        if(nullrows > 0):
            if(df[column].dtype == 'object'):
                df[column].fillna(df[column].mode(), inplace=True)
                print(column,"Filled with mode value")
                print('********************************************************')
            else:
                df[column].fillna(df[column].mean(), inplace=True)
                print(column,"Filled with mena value")
                print('********************************************************')

    df['MS SubClass'] = df['MS SubClass'].astype('object')

    print('********************************************************')
    print('One Hot Encoding')
    for column in df.columns:
        dt = df[column].dtype
        if (dt == 'object'):
            df = pd.get_dummies(df, columns=[column], drop_first=True)

    print('********************************************************')
    print('Normalization')
    

    for column in df.columns:
        dt = df[column].dtype
        if (dt == 'int64' or dt == 'float64'):
            df[column] = scaler.fit_transform(df[[column]])
                

    print(df.head())
    df.to_csv('processed_file.csv', index=False)
    return df

def split_data(df):
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, r2

#def predict(model, X):
    #y_pred_scaled = model.predict(X)
    #y = scaler.inverse_transform(y_pred_scaled)
    #return y_pred_scaled

def log_to_mlflow(model, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # Log hyper parameters using in Random Forest Algorithm
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("n_estimators", model.n_estimators)

        # Log model metrics
        y_pred = model.predict(X_test)
        MAE = mean_absolute_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)
        
        mlflow.log_metric("Mean Absolute Error", MAE)
        mlflow.log_metric("R2 Score", r2score)
    

        # Log system metrics
        # Example: CPU and Memory Usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        mlflow.log_metric("system_cpu_usage", cpu_usage)
        mlflow.log_metric("system_memory_usage", memory_usage)

        # Log execution time for training the model
        execution_time = {}  # Dictionary to store execution times for different stages
        # Example: Execution time for training the model
        start_time = time.time()
        model = train_model(X_train, y_train)
        end_time = time.time()
        execution_time["system_model_training"] = end_time - start_time

        # Log execution time 
        mlflow.log_metrics(execution_time)

        # Evaluate model and log metrics
        evaluate_model(model, X_test, y_test)

        # Log model
        mlflow.sklearn.log_model(model, "model")
        # Register the model

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_version = mlflow.register_model(model_uri, "MyModelName")


# Main function
def main():
    # Load the dataset
    data = pd.read_csv("AmesHousing.csv") 

    # Preprocess the data
    data = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
   # mae, r2 = evaluate_model(model, X_test, y_test)
   # Evaluate and log to MLflow
    log_to_mlflow(model, X_train, X_test, y_train, y_test)

    # Print the results
    #print("Mean Absolute Error:", mae)
    #print("R^2 Score:", r2)
    
    print('********************************************************')
    #data = pd.read_csv("AmesHousing.csv") 
    #data = pd.read_csv("processed_file.csv")
    #first_row = data.head(1)
    #first_row = preprocess_data(first_row)
    #first_row = first_row.drop('SalePrice', axis=1)    
    #y = predict(model, first_row)
    #print("Predicted Sale Price:", y)



if __name__ == "__main__":
    main()