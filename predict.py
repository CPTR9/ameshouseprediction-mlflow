import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
logged_model = 'runs:/2f881083e1a24dfa9f5a547c2a9fd944/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

df = pd.read_csv("processed_file.csv")
df = df.drop(['SalePrice'], axis=1)
df = df.head(3)
pred = loaded_model.predict(pd.DataFrame(df))
print(pred)