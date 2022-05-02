# Nexuni
* Collect data: 
```bash
python3.8 track.py --source path/to/video --yolo_model yolov5x.pt --classes 0 1 2 3 5 7 24 26 28 --half --save-vid --save-as-dataset
```
* Detection:
```bash
python3.8 track.py --source path/to/video --yolo_model yolov5x.pt --classes 0 1 2 3 5 7 24 26 28 --half --save-vid --hide-box --project /yolo_image
```
* Train:
```bash
python3.8 train.py --img 640 --batch 16 --epochs 3 --data ./nexuni/nexuni.yaml --weights yolov5s.pt
```
