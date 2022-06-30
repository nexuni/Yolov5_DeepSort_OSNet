# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

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
from datetime import datetime

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def on_segment(p, q, r):
    # check if r lies on (p,q)

    if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
        return True
    return False

def orientation(p, q, r):
    # return 0/1/-1 for colinear/clockwise/counterclockwise

    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0 : return 0
    return 1 if val > 0 else -1

def intersects(seg1, seg2):
    # check if seg1 and seg2 intersect
    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    # find all orientations
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
    # check general case
        return True

    if o1 == 0 and on_segment(p1, q1, p2) : return True
    # check special cases
    if o2 == 0 and on_segment(p1, q1, q2) : return True
    if o3 == 0 and on_segment(p2, q2, p1) : return True
    if o4 == 0 and on_segment(p2, q2, q1) : return True

    return False

def across_boundary(segment, boundary, reverse):
    normal = normal_vec(boundary)
    p1 = np.array(boundary[0])
    t1, t2 = np.array(segment)

    p1t1_dot_normal = float(np.sum((t1-p1) * normal)) # before
    p1t2_dot_normal = float(np.sum((t2-p1) * normal)) # after
    
    '''
    outter <=> normal vector          <=> dot product > 0
    inner  <=> negative normal vector <=> dot product < 0
    '''
    # print(p1t1_dot_normal, p1t2_dot_normal, normal)
    if p1t1_dot_normal > 0 and p1t2_dot_normal < 0: # outter to inner
        if reverse: return -1
        else: return 1
    elif p1t1_dot_normal < 0 and p1t2_dot_normal > 0: # inner to outter
        if reverse: return 1
        else: return -1
    else:
        return 0
    

def normal_vec(segment):
    p1, p2 = segment
    return np.array((p2[1] - p1[1], - (p2[0] - p1[0])))

def detect(opt, class_mapping):
    # out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
    #     project, exist_ok, update, save_crop = \
    #     opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
    #     opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop
    
    webcam = opt.source == '0' or opt.source.startswith(
        'rtsp') or opt.source.startswith('http') or opt.source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    opt.half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not opt.evaluate:
        if os.path.exists(opt.output):
            pass
            shutil.rmtree(opt.output)  # delete output folder
        os.makedirs(opt.output)  # make new output folder

    # Directories
    if type(opt.yolo_model) is str:  # single yolo model
        exp_name = opt.yolo_model.split(".")[0]
    elif type(opt.yolo_model) is list and len(opt.yolo_model) == 1:  # single models after --yolo_model
        exp_name = opt.yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + opt.deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(opt.project) / exp_name, exist_ok=opt.exist_ok)  # increment run if project name exists
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    if opt.half:
        model = DetectMultiBackend(opt.yolo_model, device=device, dnn=opt.dnn, fp16=True)
    else:
        model = DetectMultiBackend(opt.yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(opt.imgsz, s=stride)  # check image size

    # Half
    opt.half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if opt.half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(opt.source, img_size=imgsz, stride=stride, auto=pt)


    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort =  DeepSort(
        opt.deep_sort_model,
        device,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
    )

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)

    # Run tracking
    object_cache = {}
    people_counting = 2
    save_dataset_idx = 0

    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    processed_times, processed_images = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, resize_image, raw_image, vid_cap, logger_string) in enumerate(dataset):
        '''
        resize_image.shape: (3, imgsz, imgsz) resized
        raw_image.shape: (h, w, 3) original size
        '''
        t1 = time_sync()
        resize_image = torch.from_numpy(resize_image).to(device)
        resize_image = resize_image.half() if opt.half else resize_image.float()  # uint8 to fp16/32
        resize_image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(resize_image.shape) == 3:
            resize_image = resize_image[None]  # expand for batch dim
        t2 = time_sync()
        processed_times[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(resize_image, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        processed_times[1] += t3 - t2

        # Apply NMS
        [pred] = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        processed_times[2] += time_sync() - t3

        # Process detections
        processed_images += 1
        if webcam:  # nr_sources >= 1
            p = path
            p = Path(p)  # to Path
            txt_file_name = p.name
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
        else:
            p = path
            p = Path(p)  # to Path
            # video file
            if opt.source.endswith(VID_FORMATS):
                txt_file_name = p.stem
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # folder with imgs
            else:
                txt_file_name = p.parent.name  # get folder name containing current img
                save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

        logger_string += '%gx%g ' % resize_image.shape[2:]  # print string

        annotator = Annotator(raw_image, line_width=2, pil=not ascii)

        if pred is not None and len(pred):
            # Rescale boxes from img_size to raw_image size
            pred[:, :4] = scale_coords(resize_image.shape[2:], pred[:, :4], raw_image.shape).round()

            # Print results
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                logger_string += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            xywhs = xyxy2xywh(pred[:, 0:4])
            confs = pred[:, 4]
            clss = pred[:, 5]

            # pass detections to deepsort
            t4 = time_sync()
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), raw_image)
            t5 = time_sync()
            processed_times[3] += t5 - t4

            # draw boxes for visualization
            for j, (output) in enumerate(outputs):
                # output coordinates already resized back to the original size
                bboxes, object_id, object_class, object_confidence = output[0:4], int(output[4]), int(output[5]), output[6]
                obj_x_center_normalized = ((bboxes[0] + bboxes[2]) / 2) / raw_image.shape[1]
                obj_y_center_normalized = ((bboxes[1] + bboxes[3]) / 2) / raw_image.shape[0]
                obj_w_normalized = (bboxes[2] - bboxes[0]) / raw_image.shape[1]
                obj_h_normalized = (bboxes[3] - bboxes[1]) / raw_image.shape[0]

                # if not recently updated, reset the count
                current_timestamp = datetime.now().timestamp()
                if (object_id in object_cache.keys()) and \
                        object_cache[object_id]["class_name"] == "person" and \
                            (current_timestamp - object_cache[object_id]["last_update_time"]) < opt.cache_time_thres:

                    object_segment = (
                        object_cache[object_id]["last_coord"],
                        ([obj_x_center_normalized, obj_y_center_normalized])
                    )
                    # Check if object segment intersects with boundary segment
                    if intersects(object_segment, opt.segment_boundary):
                        # across_boundary returns 0 (no crossing) or +1 (increase) or -1 (decrease)
                        across_result = across_boundary(object_segment, opt.segment_boundary, opt.reverse)
                        people_counting = people_counting + across_result

                else:
                    object_cache[object_id] = {}
                    object_cache[object_id]["class_name"] = names[object_class]

                object_cache[object_id]["last_update_time"] = current_timestamp
                object_cache[object_id]["last_coord"] = (obj_x_center_normalized, obj_y_center_normalized)

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

                label = f'{object_id} {names[object_class]} {object_confidence:.2f}'
                if not opt.hide_box:
                    annotator.box_label(bboxes, label, color=colors(object_class, True))
            
            LOGGER.info(f'{logger_string}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s), People: {people_counting}')
        else:
            deepsort.increment_ages()
            LOGGER.info('No detections')


        # Stream results
        raw_image = annotator.result()

        # Save results (image with detections)
        if opt.save_vid:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 15, raw_image.shape[1], raw_image.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            scaled_back_p1 = (int(opt.segment_boundary[0][0] * w), int(opt.segment_boundary[0][1] * h))
            scaled_back_p2 = (int(opt.segment_boundary[1][0] * w), int(opt.segment_boundary[1][1] * h))
            cv2.line(raw_image,scaled_back_p1, scaled_back_p2, (0, 0, 255), 1)
            cv2.putText(raw_image,f'People: {people_counting}', 
                opt.bottomLeftCornerOfText, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                opt.fontScale,
                opt.fontColor,
                opt.fontThickness,
                opt.fontLineType)
            vid_writer.write(raw_image)
            cv2.imwrite("temp_image/temp.png", raw_image)
            # exit(-1)


    # Print results
    t = tuple(x / processed_images * 1E3 for x in processed_times)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if opt.save_vid:
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    if opt.update:
        strip_optimizer(opt.yolo_model)  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_ain_x1_0_MSMT17')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
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
    parser.add_argument('--bottomLeftCornerOfText', type=tuple, default=(0, 50))
    parser.add_argument('--fontColor', type=tuple, default=(0, 0, 255))
    parser.add_argument('--fontScale', type=int, default=2)
    parser.add_argument('--fontThickness', type=int, default=3)
    parser.add_argument('--fontLineType', type=int, default=2)
    parser.add_argument('--save-as-dataset', action='store_true', help='save the prediction results to coco dataset format')
    parser.add_argument("--save-dataset-path", type=str, default="./yolov5/nexuni/dataset")
    parser.add_argument("--save-dataset-type", type=str, default="train")
    parser.add_argument('--hide-box', action='store_true', help='hide the cropping box')
    parser.add_argument('--reverse', action='store_true', help='True: inner to outer <=> increase')
    parser.add_argument('--cache-time-thres', type=int, default=30, help='Remove object in cache if not recently updated (in UNIX time)')
    parser.add_argument('--segment-boundary', type=tuple, default=((0.4, 0.3), (0.6, 0.57)), help='normalized ((x1, y1), (x2, y2)')
    opt = parser.parse_args()

    # Forcing p1x <= p2x
    p1, p2 = opt.segment_boundary
    if p1[0] >= p2[0]:
        p1, p2 = p2, p1
        opt.segment_boundary = (p1, p2)

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

    with torch.no_grad():
        detect(opt, class_mapping)
