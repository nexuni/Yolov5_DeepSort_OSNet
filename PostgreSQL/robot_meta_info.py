import requests
from PostgreSQL.schema import *

def add_robot_meta_info(body):
    body["service"] = "robot_meta_info"
    body["operation"] = "add_robot_meta_info"
    return requests.post(DB, json=body)

def get_robot_meta_info(params):
    params["service"] = "robot_meta_info"
    params["operation"] = "get_robot_meta_info"
    return requests.get(DB, params=params).json()["message"]

def update_robot_meta_info(body):
    body["service"] = "robot_meta_info"
    body["operation"] = "update_robot_meta_info"
    return requests.post(DB, json=body)