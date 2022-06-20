#!/usr/bin/env python
# coding: utf-8

# In[14]:


from azureml.core import Workspace
ws = Workspace.from_config()


# In[15]:


from azureml.core import Experiment
experiment = Experiment(workspace=ws, name="diabetes-experiment")


# In[16]:


from azureml.opendatasets import Diabetes
from sklearn.model_selection import train_test_split

x_df = Diabetes.get_tabular_dataset().to_pandas_dataframe().dropna()
y_df = x_df.pop("Y")

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=66)


# In[17]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import math

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for alpha in alphas:
    run = experiment.start_logging()
    run.log("alpha_value", alpha)
    
    model = Ridge(alpha=alpha)
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X=X_test)
    rmse = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    run.log("rmse", rmse)
    
    model_name = "model_alpha_" + str(alpha) + ".pkl"
    filename = "outputs/" + model_name
    
    joblib.dump(value=model, filename=filename)
    run.upload_file(name=model_name, path_or_stream=filename)
    run.complete()


# In[18]:


experiment


# In[19]:


minimum_rmse_runid = None
minimum_rmse = None

for run in experiment.get_runs():
    run_metrics = run.get_metrics()
    run_details = run.get_details()
    # each logged metric becomes a key in this returned dict
    run_rmse = run_metrics["rmse"]
    run_id = run_details["runId"]
    
    if minimum_rmse is None:
        minimum_rmse = run_rmse
        minimum_rmse_runid = run_id
    else:
        if run_rmse < minimum_rmse:
            minimum_rmse = run_rmse
            minimum_rmse_runid = run_id

print("Best run_id: " + minimum_rmse_runid)
print("Best run_id rmse: " + str(minimum_rmse))    


# In[20]:


from azureml.core import Run
best_run = Run(experiment=experiment, run_id=minimum_rmse_runid)
print(best_run.get_file_names())


# In[21]:


best_run.download_file(name="model_alpha_0.1.pkl")


# In[22]:


model = best_run.register_model(model_name='diabetes-model', 
                                model_path='model_alpha_0.1.pkl')


# In[23]:


get_ipython().run_cell_magic('writefile', 'score.py', 'import json\nimport numpy as np\nimport os\nimport pickle\nfrom sklearn.externals import joblib\nfrom sklearn.linear_model import Ridge\nfrom sklearn.linear_model import LogisticRegression\nfrom azureml.core.model import Model\n\ndef init():\n    global model\n    # retrieve the path to the model file using the model name\n    model_path = Model.get_model_path(\'model_alpha_0.1.pkl\')   \n    model = joblib.load(model_path)\n\ndef run(raw_data):\n    try:\n        data = np.array(json.loads(raw_data)[\'data\'])\n        # make prediction\n        y_hat = model.predict(data)\n        # you can return any data type as long as it is JSON-serializable\n        return y_hat.tolist()\n    except Exception as e:\n        result = str(e)\n        # return error message back to the client\n        return json.dumps({"error": result})\n')


# In[24]:


from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.image import ContainerImage

# Create an empty conda environment and add the scikit-learn package
env = CondaDependencies()
env.add_conda_package("scikit-learn")

# Display the environment
print(env.serialize_to_string())

# Write the environment to disk
with open("myenv.yml","w") as f:
    f.write(env.serialize_to_string())


# In[25]:


from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={'data': 'AML notebook - diabetes'}, 
                                               description='Predict people at risk diabetes')

