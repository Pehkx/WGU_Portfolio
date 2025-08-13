#!/usr/bin/env python
# coding: utf-8

# Import statements

# In[3]:


# Importing Packages
from fastapi import FastAPI, HTTPException # For running API
import uvicorn # For environment for API
import json # To import and manipulate json files
import numpy as np # For basic functionality and passing into an array
import pickle # To import and analyze .pkl files
import datetime # For delay calculations via date and timestamps


# To-Do List

# In[5]:


# TODO:  
# 1) write the back-end logic to provide a prediction given the inputs
# 2) requires finalized_model.pkl to be loaded
# 3) the model must be passed a NumPy array consisting of the following:
# (polynomial order, encoded airport array, departure time as seconds since midnight, arrival time as seconds since midnight)
# the polynomial order is 1 unless you changed it during model training in Task 2
# 4) write the API Endpoints


# Opening arrival airport list

# In[7]:


# Opening airport encodings 
with open('airport_encodings.json', 'r') as f: # Updated to use with and "r" to ensure read only
    airports = json.load(f)

# load trained model
with open ("finalized_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
    ## Satisfies 2) above


# In[8]:


def create_airport_encoding(airport: str, airports: dict) -> np.array:
    
    """
    create_airport_encoding is a function that creates an array the length of all arrival airports from the chosen
    departure aiport.  The array consists of all zeros except for the specified arrival airport, which is a 1.  

    Parameters
    ----------
    airport : str
        The specified arrival airport code as a string
    airports: dict
        A dictionary containing all of the arrival airport codes served from the chosen departure airport
        
    Returns
    -------
    np.array
        A NumPy array the length of the number of arrival airports.  All zeros except for a single 1 
        denoting the arrival airport.  Returns None if arrival airport is not found in the input list.
        This is a one-hot encoded airport array.

    """
    
    temp = np.zeros(len(airports))
    if airport in airports:
        temp[airports.get(airport)] = 1
        return temp
    else:
        return None


# In[9]:


# Create FastAPI instance
app = FastAPI()

# Creating function to convert time to seconds
def convert_time_to_seconds(time_str: str) -> int:
    try:
        h, m = map(int, time_str.split(":"))
        if not (0 <=h <24 and 0 <= m < 60): # Update to check for valid time range for unit testing
            return None
        return h * 3600 + m * 60
    except ValueError:
        return None

# Endpoint to check if API is functional
## Satisfies B1 of the Rubric above
@app.get("/")
def root():
    return {"message": "Airport Delay Prediction API is running"}

# Endpoint to predict airport departure delays
## Satisfies B2 of the Rubric
@app.get("/predict/delays")
def predict_delays(arrival_airport: str, departure_time: str, arrival_time: str):
    
    # Validation and Error Codes
    ## To setup F of the Rubric
    one_hot_airport = create_airport_encoding(arrival_airport, airports) 
    if one_hot_airport is None: 
        raise HTTPException(status_code=400, detail="Invalid arrival airport code.") # Validating correct airport codes

    departure_time_seconds = convert_time_to_seconds(departure_time) 
    arrival_time_seconds = convert_time_to_seconds(arrival_time)
    if departure_time_seconds is None or arrival_time_seconds is None: 
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM.") # Validating correct time format

    # Prepare Input Data
    input_data = np.hstack(([1], one_hot_airport, [departure_time_seconds, arrival_time_seconds]))

    # Make Prediction
    prediction = model.predict(input_data.reshape(1, -1))[0]

    return {"predicted_delay": round(float(prediction), 2)} 


# In[ ]:


# Updated to run properly in a Docker image
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


