import requests
from PostgreSQL.schema import *


def get_anomaly(params):
    params["service"] = "anomaly"
    params["operation"] = "get_anomaly"
    return requests.get(DB, params=params).json()["message"]

def add_anomaly(body):
    body["service"] = "anomaly"
    body["operation"] = "add_anomaly"
    return requests.post(DB, json=body)

def update_anomaly(body):
    body["service"] = "anomaly"
    body["operation"] = "update_anomaly"
    return requests.post(DB, json=body)