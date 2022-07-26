from elasticsearch import Elasticsearch
from tqdm import tqdm


class ElasticSearchResult:
    def __init__(self, cloud_id, elastic_user, elastic_password):
        self._INDEX = "docs"
        self.client = Elasticsearch(cloud_id=cloud_id, basic_auth=(elastic_user, elastic_password))

    def indexing(self, document_list):
        i = 0
        for doc in document_list:
            ans = self.client.index(index=self._INDEX, id=i, document=doc)
            i+=1
    

    def get_index(self):
        print(self.client.indices.get(index=self._INDEX))
        print('done')
    def delete_doc(self):
        self.client.indices.delete(index=self._INDEX, ignore=[400, 404])
    
    def search(self, query, k: int=10, _from: int=0):
        body = {
            'from' : 0,
            'size' : k,
            'query': {
                'match': {
                    'paragraphs': query
                }
            }
        }
        out = self.client.search(index=self._INDEX, body=body)
        return out