from os import stat
from fastapi import FastAPI
import ir_system
import elastic_search

app = FastAPI()
initial = ir_system.Initial()
# initial_elastic = elastic_search.ElasticsearchResult()


@app.get('/')
def index():
    return 'connected!'


@app.get('/result')
def get_query_result(query: str, action_type: str, query_expand: bool):
    return initial.find_target(query, action_type, query_expand)
