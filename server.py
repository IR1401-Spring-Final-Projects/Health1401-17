from os import stat
from fastapi import FastAPI
import ir_system
import elastic_search

app = FastAPI()
initial = ir_system.Initial()

@app.get('/')
def index():
    return 'connected!'

@app.get('/result')
def get_query_result(query:str, action_type:str):
    return initial.find_target(query, action_type)