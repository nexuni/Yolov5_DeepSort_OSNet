import requests
from PostgreSQL.schema import *

def get_task(params):
    params["service"] = "task"
    params["operation"] = "get_task"
    return requests.get(DB, params=params).json()["message"]

def add_task(body):
    body["service"] = "task"
    body["operation"] = "add_task"
    return requests.post(DB, json=body)

def update_task(body):
    body["service"] = "task"
    body["operation"] = "update_task"
    return requests.post(DB, json=body)