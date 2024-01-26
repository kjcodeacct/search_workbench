import pandas as pd

class ESHandler:
    def __init__(self, esClient, indexName):
        self.esClient = esClient
        self.indexName = indexName


    def buildQuery(self, input):
        # elasticSearchQuery = {
        #     'size' : 25,
        #     'query': {
        #     'match_all' : {}
        #     }
        # }

        knnQuery = {
            "field": "phrase-vector",
            "query_vector": input,
            "k": 10,
            "num_candidates": 25
        }

        return knnQuery

    def runQuery(self, query):
        # response = self.esClient.search(
        #     index=self.indexName, body = query
        # )   

        response = self.esClient.knn_search(
            index=self.indexName
            , knn=query 
            , source=["id","name", "lat", "lon", "location", "country", "population"]
        )

        return response
    
    def responseToDF(self,response):

        jsonResults = []
        for hit in response["hits"]["hits"]:
            jsonResults.append(hit["_source"])

        df = pd.DataFrame.from_records(jsonResults)
        # df = json_normalize(jsonResults)
        return df