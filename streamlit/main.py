import streamlit as st
from streamlit_folium import st_folium
from shapely import wkt
import pandas as pd
import plotly.express as px
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import numpy as np
from elasticsearch import Elasticsearch
from es_handler import ESHandler
import spacy
import spacy_streamlit

import gc
import torch

# import spacy_sentence_bert
from utils import prettyJson, placeHolder, jsonDictToDF, processLocationGeom, processGeoGeom, modelDir


# init
gc.collect()
torch.cuda.empty_cache()

appName = "Search Workbench"
st.set_page_config(page_title=appName, layout="wide")
# st.title(appName)

if 'searchInput' not in st.session_state:
    st.session_state['searchInput'] = ''

esClient = Elasticsearch("http://localhost:9200")
esHandler = ESHandler(esClient=esClient, indexName="knn-index")

# load one of the models listed at https://github.com/MartinoMensio/spacy-sentence-bert/
nlp = spacy.load(modelDir()) # vector dimensions of 1024

df = None
nlpDoc = None
pd.options.plotting.backend = "plotly"

# sidebar

with st.sidebar:
    st.text_input("Search:",  key="searchInput")

    st.text("Results:")
    if st.session_state.searchInput != "":
        nlpDoc=nlp(st.session_state.searchInput)
        esQuery = esHandler.buildQuery(nlpDoc.vector)
        response = esHandler.runQuery(esQuery)
        df = esHandler.responseToDF(response)

        resultSummaryDF = df[['name', 'id']]

        st.dataframe(data=resultSummaryDF)

    st.divider()

    with st.expander("Query Debugger"):
        # if st.session_state.searchInput != "":
            # nlpDoc=nlp(st.session_state.searchInput)
        
            # esQuery = esHandler.buildQuery(nlpDoc.vector)
            # st.code(prettyJson(esQuery), language="json", line_numbers=False)
            # if st.button(label='Run' , type="primary", key="esRun"):
            #     st.text("ran request...")
           
        # else:
        st.code("""{"foo":"bar"}""", language="json", line_numbers=False)

    with st.expander("HTTP Debugger"):
            st.code("""{"foo":"bar"}""", language="json", line_numbers=False)

    st.divider()


# main content tabs

st.header("Search Query: "+ st.session_state.searchInput)
with st.expander("Text Analysis"):
    with st.container():
        if st.session_state.searchInput != "":
            analysisNLP = spacy.load("en_core_web_sm")
            spacy_streamlit.visualize_parser(analysisNLP(st.session_state.searchInput))
        else:
            placeHolder()


with st.expander("Map Results", expanded=True):
    with st.container():
        mapCol, mapClickCol= st.columns(spec=[0.6, 0.4], gap="medium")

        with mapCol:
            st.session_state["mapCenter"] =[39.81060314270684, -98.55426793012649,]
            st.session_state["mapZoom"] = 4

            mapView = folium.Map(location=st.session_state["mapCenter"], zoom_start=st.session_state["mapZoom"])

            if st.session_state.searchInput != "":

                df["processedLocation"] = df.apply(processLocationGeom, axis=1)
                gdf = gpd.GeoDataFrame(df, geometry="processedLocation")
                gdf.set_crs(epsg=4326, inplace=True)

                popup = folium.GeoJsonPopup(
                    fields=["name"],
                    aliases=[""],
                    localize=True,
                    # labels=True,
                    style="background-color: yellow;",
                )

                folium.GeoJson(data=gdf, popup=popup).add_to(mapView)

            st.text("‎")
            mapDataView = st_folium(
                mapView,
                center=st.session_state["mapCenter"],
                zoom=st.session_state["mapZoom"],
                key="mapView",
                use_container_width=True,
                height=400,
            )

        with mapClickCol:
            with st.container():
                st.text("Selected Map Result:")
                if mapDataView["last_active_drawing"]:
                        clickedResultDf = pd.DataFrame([mapDataView["last_active_drawing"]["properties"]])
                        clickedResultDf = clickedResultDf.transpose()
                        clickedResultDf.columns = ['Result']
                        st.dataframe(data=clickedResultDf, use_container_width=True, height=400)           

with st.expander("Graphs"):
    with st.container():

        row1Graph1, row1Graph2= st.columns(spec=[0.5, 0.5], gap="medium")
        row2Graph1, row2Graph2= st.columns(spec=[0.5, 0.5], gap="medium")

        if st.session_state.searchInput != "":
            with row1Graph1:
                fig = px.pie(df.query("population>0"), values='population', names='country')
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            with row1Graph2:
                # docs on scatter geo type
                # https://plotly.com/python-api-reference/generated/plotly.express.html#plotly.express.scatter_geo
                fig = px.scatter_geo(
                    df.query("population>0"), lat="lat", lon="lon", color="country",
                            hover_name="name", size="population",
                             projection="natural earth")

                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            with row2Graph1:
                st.text("Cities per Country")
                countryDf = df.filter(['country'], axis=1)
                fig = countryDf.plot.hist()
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            with row2Graph2:
                st.text("Country X Population")
                populationDf = df.filter(['country','population'], axis=1)
                # fig = populationDf.plot.hist()
                fig = px.scatter(
                    df.query("population>0"),
                    x="country",
                    y="population",
                    color="population",
                    color_continuous_scale="reds",
                )
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        else:
            placeHolder()

with st.expander("Result List", expanded=True):
    with st.container():
        if st.session_state.searchInput != "":
            st.dataframe(data=df, use_container_width=True, height=400)
        else:
            placeHolder()

# textAnalysisTab, resultTab, mapTab, = st.tabs(["Text Analysis", "Results", "Map"])

# with textAnalysisTab:
#     if st.session_state.searchInput != "":
#         # https://spacy.io/universe/project/spacy-sentence-bert
#         # using one of these models specified here
#         models = ["en_stsb_roberta_large"]
#         default_text = st.session_state.searchInput
#         # spacy_streamlit.visualize(models, default_text)
#     else:
#         placeHolder()

# with resultTab:
#     if st.session_state.searchInput != "":
#         st.dataframe(data=df)
#     else:
#         placeHolder()



# with mapTab:
    
   