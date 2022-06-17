import socket
import time
from datetime import datetime
import cv2
import os
import json
from pathlib import Path

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
HOST = '127.0.0.1'
PORT = 8000
server_addr = (HOST, PORT)
video_path = "onecar.MOV"
image_dir = "./images"
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
if not video.isOpened():
    print(f"Fail to open {video_path}")

success = True
frame_idx = 0
while success:
    success, frame = video.read()
    if not success:
        break
    filepath = str((Path(image_dir) / f'{str(frame_idx).zfill(8)}.png').resolve())
    data = json.dumps({
        "type": "cross",
        "filepath": filepath,
        "timestamp": datetime.now().timestamp()
    })
    print(f"Send: {data}, Type: {type(data)}")
    cv2.imwrite(filepath,frame)
    client.sendto(data.encode(), server_addr)
    frame_idx+=1

client.sendto(json.dumps({"type": "END", "filepath": "", "timestamp": datetime.now().timestamp()}).encode(), server_addr)
