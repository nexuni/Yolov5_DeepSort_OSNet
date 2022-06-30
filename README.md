# Usage
* Anomaly detection
```bash
python3.8 track.py --source path/to/video --half --save-vid --hide-box --project /image --classes 0 1 2 3 5 7 24 26 28 --robot-id NSR-0
```
* Patrol (fire hydrant & cone)
```bash
python3.8 track_udp.py --half --save-vid --project /image --classes 10
```
* People flow counting
```bash
python3.8 track_people_count_segment.py --source path/to/video --half --save-vid --project /image --classes 0 
```
* Train:
```bash
python3.8 train.py --img 640 --batch 16 --epochs 3 --data ./nexuni/nexuni.yaml --weights yolov5x.pt
```

# Parameter Tuning
* DeepSort (deep_sort/configs/deep_sort.yaml)  
`MAX_DIST`: The matching threshold. Samples with larger distance are considered an invalid match  
`MAX_IOU_DISTANCE`: Gating threshold. Associations with cost larger than this value are disregarded.  
`MAX_AGE`: Maximum number of missed misses before a track is deleted  
`N_INIT`: Number of frames that a track remains in initialization phase  
`NN_BUDGET`: Maximum size of the appearance descriptors gallery  
* Tracking (track_udp.py, track_people_count_segment.py)  
General  
`cache-time-thres`: Remove object in cache if not recently updated  
`image-timestamp-thres`: Skip the image if out of date  
Anomaly  
`anomalies`: Define the anomaly  
`anomaly-save-thres`: Save the crop image if detected id count > threshold  
Cross road  
`vehicles`: Define the vehicles  
`cross-road-frame-thres`: Determine cross or deny after detecting cross-road-frame-thres images  
`displacement-thres`: Consider vehicles moving if displacement > displacement-thres  
`moving-vehicles-thres`: Determine cross or deny if moving vehicles > moving-vehicles-thres  
Patrol  
`landmark`: Define the patrol landmark  
`landmark-frame-thres`: Save the crop image if detected id exceed the threshold  
People flow counting  
`reverse`: True: inner to outer <=> increase | False:  outer to inner <=> increase  
`segment-boundary`: Define the crossing boundary by 2 coordinates (normalized, (0,0) at top left)  
    
# Custom dataset
* [Yolov5 training (link to external repository)](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp;
