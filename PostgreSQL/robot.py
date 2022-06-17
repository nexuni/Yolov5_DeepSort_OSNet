from zlib import DEF_BUF_SIZE
import requests
from PostgreSQL.schema import *

def get_robot(params):
    params["service"] = "robot"
    params["operation"] = "get_robot"
    return requests.get(DB, params=params).json()["message"]

def add_robot(body):
    body["service"] = "robot"
    body["operation"] = "add_robot"
    return requests.post(DB, json=body)

def update_robot(body):
    body["service"] = "robot"
    body["operation"] = "update_robot"
    return requests.post(DB, json=body)
