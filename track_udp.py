# # limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import glob
import argparse
import yaml
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.augmentations import letterbox
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

from PostgreSQL.schema import *
from PostgreSQL.anomaly import *
from datetime import datetime
import socket

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import json
import socket
import threading
import time
import queue

data_queue = queue.Queue()

def Collect_data():
    HOST = '127.0.0.1'
    PORT = 8000
    server_addr = (HOST, PORT)
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(server_addr)
    print("Data thread running")
    while True:
        try:
            indata, addr = server.recvfrom(1024)
            indata = json.loads(indata.decode())
            # print(f"receive data: {indata}")
            data_queue.put(indata)
        except Exception as e:
            print(e)
        
def test_detect():
    fps, size = 15, (1280, 720)
    writer = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    q = []
    while True:
        try:
            data = data_queue.get()
            print(f"processing data: {data}")
            writer.write(cv2.imread(data["filepath"]))
        except:
            pass  

def detect(opt, class_mapping):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
        project, exist_ok, update, save_crop = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = f"{exp_name}_{deep_sort_model.split('/')[-1].split('.')[0]}"
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn, fp16=True) if half else DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort =  DeepSort(
        deep_sort_model,
        device,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
    )

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run tracking
    object_cache = {}
    vehicles = set(['bicycle', 'car', 'motorcycle', 'bus', 'truck'])
    moving_vehicles_idx = 0
    cross_road_frame_cnt = 0
    save_dataset_idx = 0
    video_writer = None

    model.warmup(imgsz=(1, 3, *imgsz))
    processed_times, processed_images = [0.0, 0.0, 0.0, 0.0], 0
    while True:
        try:
            '''
            data: {"type": ..., "filepath": ..., "timestamp": ...}
            im.shape: (3, imgsz, imgsz) resized
            raw_image.shape: (h, w, 3) original size
            '''
            data = data_queue.get()
            data_path, data_type, data_timestamp = data["filepath"], data["type"], data["timestamp"]

            if (datetime.now().timestamp() - data_timestamp) > opt.image_timestamp_thres:
                continue

            if data_type == "cross":
                cross_road_frame_cnt +=1
            elif data_type == "anomaly":
                cross_road_frame_cnt = 0
            else:
                break
            raw_image = cv2.imread(data_path)
            im = letterbox(raw_image, imgsz, stride=stride, auto=pt)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)
            logger_string = f"{(data_path)}: "
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        # Preprocessing image
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        processed_times[0] += t2 - t1

        # Inference
        pred = model(im, augment=opt.augment, visualize=False)
        t3 = time_sync()
        processed_times[1] += t3 - t2

        # Apply NMS
        [pred] = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        processed_times[2] += time_sync() - t3

        # Processing detections
        processed_images += 1
        p = Path(data_path)
        save_path = str(save_dir / "result")  # im.jpg, vid.mp4, ...

        logger_string += '%gx%g ' % im.shape[2:]

        annotator = Annotator(raw_image, line_width=2, pil=not ascii)

        if pred is not None and len(pred):
            # Rescale boxes from img_size to raw_image size
            pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], raw_image.shape).round()

            # Print results
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                logger_string += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            xywhs = xyxy2xywh(pred[:, 0:4])
            confs = pred[:, 4]
            clss = pred[:, 5]

            # Pass detections to deepsort
            t4 = time_sync()
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), raw_image)
            t5 = time_sync()
            processed_times[3] += t5 - t4

            # Draw boxes for visualization
            for j, (output) in enumerate(outputs):
                save_crop = False
                # output coordinates already resized back to the original size
                bboxes, object_id, object_class, object_confidence = output[0:4], int(output[4]), int(output[5]), output[6]
                obj_x_center_normalized = ((bboxes[0] + bboxes[2]) / 2) / raw_image.shape[1]
                obj_y_center_normalized = ((bboxes[1] + bboxes[3]) / 2) / raw_image.shape[0]
                obj_w_normalized = (bboxes[2] - bboxes[0]) / raw_image.shape[1]
                obj_h_normalized = (bboxes[3] - bboxes[1]) / raw_image.shape[0]

                # Increase object count; if not recently updated, reset the count
                current_timestamp = datetime.now().timestamp()
                if (object_id in object_cache.keys()) and \
                        (current_timestamp - object_cache[object_id]["last_update_time"]) < opt.cache_time_thres:
                    object_cache[object_id]["count"] +=1
                else:
                    object_cache[object_id] = {}
                    object_cache[object_id]["count"] = 1
                    object_cache[object_id]["first_coord"] = np.array([obj_x_center_normalized, obj_y_center_normalized])
                    object_cache[object_id]["class_name"] = names[object_class]

                object_cache[object_id]["last_update_time"] = current_timestamp
                object_cache[object_id]["last_coord"] = np.array([obj_x_center_normalized, obj_y_center_normalized])

                # For anomaly
                if not (object_cache[object_id]["class_name"] in vehicles):
                    if object_cache[object_id]["count"] > opt.save_thres:
                        save_crop = True
                        del object_cache[object_id]

                # For dataset collecting
                if opt.save_as_dataset:
                    # Save image
                    cv2.imwrite(os.path.join(
                        opt.save_dataset_path,
                        "images", opt.save_dataset_type,
                        f"{str(save_dataset_idx).zfill(10)}.png"), raw_image.copy())

                    # Save label
                    _class = class_mapping[names[object_class]]
                    with open(os.path.join(
                        opt.save_dataset_path,
                        "labels", opt.save_dataset_type,
                        f"{str(save_dataset_idx).zfill(10)}.txt"), 'a') as f: 
                        f.write(f"{_class} {obj_x_center_normalized} {obj_y_center_normalized} {obj_w_normalized} {obj_h_normalized}\n")
                    
                    save_dataset_idx+=1

                # Add bbox to image
                if save_vid or save_crop:
                    label = f'{object_id} {names[object_class]} {object_confidence:.2f}'
                    if not opt.hide_box:
                        annotator.box_label(bboxes, label, color=colors(object_class, True))
                    if save_crop:
                        LOGGER.info(f"Save: {names[object_class]}")
                        _, full_path = save_one_box(
                            bboxes,
                            raw_image.copy(),
                            file=Path(os.path.join(
                                save_dir,'crops', names[object_class], f'id_{object_id}', f'{p.stem}.jpg')),
                            BGR=True
                            )
                        add_anomaly({
                            ANOMALY_TYPE: names[object_class],
                            ROBOT_ID: opt.robot_id,
                            SITE_ID: "site A",
                            ANOMALY_LAT: "24.987292967314355",
                            ANOMALY_LNG: "121.5522044067383",
                            ANOMALY_IMAGE_PATH: 'asset' + full_path,
                            TIME: datetime.now().timestamp()
                        })
            
            # Determine cross road
            if cross_road_frame_cnt > opt.cross_road_frame_thres:
                moving_vehicles = 0
                # Iterate through all vehicles in object cache
                for obj_id, obj_info in object_cache.items():
                    # Skip if not vehicles
                    if not (object_cache[obj_id]["class_name"] in vehicles): continue
                    # Calculate displacement norm
                    displacement = object_cache[obj_id]["last_coord"] - object_cache[obj_id]["first_coord"]
                    displacement_norm = np.sqrt(np.sum(displacement**2))
                    # Consider it moving if > threshold
                    if displacement_norm > opt.displacement_thres:
                        LOGGER.info(f"Moving class: {object_cache[obj_id]['class_name']} | Moving id: {obj_id} | Norm: {displacement_norm}")
                        moving_vehicles+=1
                    # Reset the first coordinate
                    object_cache[obj_id]["first_coord"] = object_cache[obj_id]["last_coord"]

                # Stop if moving vehicles > threshold
                if moving_vehicles > opt.moving_vehicles_thres:
                    LOGGER.info("STOP")
                    cv2.imwrite(f"temp_image/{moving_vehicles_idx}.png", raw_image)
                    moving_vehicles_idx+=1
                else:
                    LOGGER.info("PASS")
                cross_road_frame_cnt = 0
            
            # Summary logger
            LOGGER.info(f'{logger_string}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
        else:
            deepsort.increment_ages()
            LOGGER.info('No detections')

        # Stream results
        raw_image = annotator.result()
        if save_vid:
            if not video_writer:
                fps, size = opt.fps, (raw_image.shape[1], raw_image.shape[0])
                video_writer = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            video_writer.write(raw_image)
        
        # Remove image after inferenced
        os.remove(data_path)

    # Print results
    t = tuple(x / processed_images * 1E3 for x in processed_times)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='resnet50')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # Custom
    parser.add_argument('--fps', type=float, default=15., help='fps for saving vedio')
    parser.add_argument('--hide-box', action='store_true', help='hide the cropping box')
    parser.add_argument('--save-as-dataset', action='store_true', help='save the prediction results to coco dataset format')
    parser.add_argument("--save-dataset-path", type=str, default="./yolov5/nexuni/dataset")
    parser.add_argument("--save-dataset-type", type=str, default="train")
    parser.add_argument("--robot-id", type=str, default="robot id")
    parser.add_argument('--save-thres', type=int, default=100, help='Save the crop image if detected id exceed the threshold')
    parser.add_argument('--cache-time-thres', type=int, default=300, help='Remove object in cache if not recently updated')
    parser.add_argument('--image-timestamp-thres', type=int, default=30, help='Skip the image if out of date')
    parser.add_argument('--cross-road-frame-thres', type=int, default=100, help='Determine cross or deny after detecting cross-road-frame-thres images')
    parser.add_argument('--displacement-thres', type=float, default=0.33, help='Consider vehicles moving if displacement > displacement-thres')
    parser.add_argument('--moving-vehicles-thres', type=int, default=0, help='Determine cross or deny if moving vehicles > moving-vehicles-thres')
    opt = parser.parse_args()


    class_mapping = None # This is for making training dataset
    if opt.save_as_dataset:
        if not os.path.exists(os.path.join(opt.save_dataset_path, "images", opt.save_dataset_type)):
            os.makedirs(os.path.join(opt.save_dataset_path, "images", opt.save_dataset_type))
        if not os.path.exists(os.path.join(opt.save_dataset_path, "labels", opt.save_dataset_type)):
            os.makedirs(os.path.join(opt.save_dataset_path, "labels", opt.save_dataset_type))
        with open("./yolov5/nexuni/nexuni.yaml", 'r') as f:
            data_format = yaml.load(f, Loader=yaml.FullLoader)
            class_mapping = {name:i for i, name in enumerate(data_format["names"])}

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    t = threading.Thread(target = Collect_data)
    t.daemon = True
    t.start()
    # test_detect()
    with torch.no_grad():
        detect(opt, class_mapping)
