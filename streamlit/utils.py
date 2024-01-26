import json
import streamlit as st
import pandas as pd
from shapely.geometry import shape
import os


def prettyJson(input):
    return json.dumps(
    input,
    indent=4,
    separators=(',', ': ')
    )

def jsonDictToDF(input):
    # data = json.loads(input)
    return pd.json_normalize(input)

def placeHolder():
    return st.text("No Current Data...")

def processLocationGeom(row):
    if row["location"]:
        return shape(row["location"])
    
def processGeoGeom(row):
    if row["geometry"]:
        return shape(row["geometry"])

def modelDir():
    modelDir = ""
    try:
        modelDir = os.environ['MODEL_DIR']
        if modelDir == "":
            modelDir = "model/en_stsb_roberta_large/en_stsb_roberta_large-0.1.2"
    except:
        modelDir = "model/en_stsb_roberta_large/en_stsb_roberta_large-0.1.2"
    
    return modelDir
