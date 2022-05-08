import socket
import time
import cv2
import os
from pathlib import Path

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
HOST = '127.0.0.1'
PORT = 8000
server_addr = (HOST, PORT)
video_path = "0.mp4"
image_dir = "./images"
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print(f"Fail to open {video_path}")

success = True
frame_idx = 0
while success:
    success, frame = video.read()
    if not success:
        break
    image_path = str((Path(image_dir) / f'{str(frame_idx).zfill(8)}.png').resolve())
    print(f"Send: {image_path}")
    cv2.imwrite(image_path,frame)
    client.sendto(image_path.encode(), server_addr)
    frame_idx+=1