#! /usr/bin/env python3

from __future__ import division
from datetime import datetime

import os
import argparse
import numpy as np
import cv2
import time
import torch
import torchvision.transforms as transforms
from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import winsound

import matplotlib
matplotlib.use('Agg')
f_save_video = False


def basketball_realtime(model, img_size, output_path, conf_thresh, nms_thresh, f_save_imgs):
    global f_save_video
    cap = cv2.VideoCapture(1, cv2.CAP_ANY)
    cap.set(3, 1920)
    cap.set(4, 1080)

    start_time_display = time.time()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-time-%H-%M-%S")
    only_date = now.strftime("%Y-%m-%d")
    img_no_base = 1000000

    if f_save_video:
        video_name = dt_string + '_output_video.mp4'
        output_video = os.path.join(output_path, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        fps = 20  # Frames per second
        video = cv2.VideoWriter(output_video, fourcc, fps, (1920, 1080))

    # Create nested directories
    if f_save_imgs:
        # # make the output dir
        img_dt_string = "Image_" + dt_string
        datapath = os.path.join(output_path, img_dt_string)
        try:
            os.makedirs(datapath)
            print(f"Nested directories '{datapath}' created successfully.")
        except FileExistsError:
            print(f"One or more directories in '{datapath}' already exist.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{datapath}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

        dt_string_roi = 'ROI_' + dt_string
        roipath = os.path.join(output_path, dt_string_roi)
        # Create nested directories
        try:
            os.makedirs(roipath)
            print(f"Nested directories '{roipath}' created successfully.")
        except FileExistsError:
            print(f"One or more directories in '{roipath}' already exist.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{roipath}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    while True:

        start_time = time.time()

        img_no_base += 1
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        image = frame[215:465, 800:1120]  # basket is at (960, 340), wanna make ROI 320 X 250
        pass_num, total_num = detect_realtime_img(model, image, img_size, conf_thresh, nms_thresh)
        end_time = time.time()
        duration = end_time - start_time_display

        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Add text to the image
        mins  = int(duration / 60)
        secs = int(duration - mins * 60)
        text1 = f" {mins}:{secs}"
        fontScale = 1
        thickness = 2
        cv2.putText(frame, text1, (10, 50), font, fontScale, (0, 255, 255), thickness, cv2.LINE_AA)
        text2 = f"{pass_num} / {total_num}"
        fontScale = 2
        thickness = 4
        cv2.putText(frame, text2, (880, 900), font, fontScale, (255, 255, 0), thickness, cv2.LINE_AA)
        text4 = "pass      total"
        fontScale = 1
        thickness = 2
        cv2.putText(frame, text4, (865, 930), font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
        text3 = only_date
        fontScale = 1
        thickness = 2
        cv2.putText(frame, text3, (1700, 50), font, fontScale, (255, 0, 255), thickness, cv2.LINE_AA)

        cv2.imshow('Basketball Shooting Match', frame)

        # next, try to save img for debugging and training purpose.
        if f_save_imgs:
            output = os.path.join(datapath, f"basketball_{dt_string}_{img_no_base}.png")
            cv2.imwrite(output, frame)  # save whole images for video making.
            output = os.path.join(roipath, f"basketball_{dt_string}_{img_no_base}.png")
            cv2.imwrite(output, image)  # save roi image for debugging and training

        if f_save_video:
            video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    cap.release()
    cv2.destroyAllWindows()
    video.release()
    print(f"Video saved as {output_video}")


def detect_realtime_img(model, image, img_size, conf_thresh, nms_thresh):
    dets = detect_image(model, image, img_size, conf_thresh, nms_thresh)
    f_result = checkEvent(dets)
    passnum, totalnum = count_pass_fail(f_result)
    return passnum, totalnum


def checkEvent(dets):
    global f_save_video
    if not hasattr(checkEvent, "confident"):
        checkEvent.confident = 0
        checkEvent.event_st = False
        checkEvent.no_bb_cnt = 0
        checkEvent.aim_good = False
        checkEvent.last_ball_posn_y = 100

    basket_x = 160
    basket_y = 125
    k_margin_x = 30  # this is subject to tuning
    k_margin_y = 30
    k_min_score = 0.5
    k_bob_x_margin = 45
    k_bob_y_margin = 40
    classes = ["ball", "bib", "bob"]
    if len(dets) > 0:
        checkEvent.no_bb_cnt = 0
        for i in range(len(dets)):
            x1, y1, x2, y2, conf, cls_pred = dets[i]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if conf > 0.5 and classes[int(cls_pred)] == "ball":
                checkEvent.last_ball_posn_y = cy
                if cy < basket_y and abs(cx - basket_x) < 100 and not checkEvent.event_st:
                    checkEvent.event_st = True
                    checkEvent.confident += 0.1
                elif not checkEvent.aim_good and abs(cx - basket_x) < k_margin_x \
                        and abs(cy - basket_y) < k_margin_y:
                    checkEvent.confident += 0.2
                    checkEvent.aim_good = True
                # elif checkEvent.event_st and checkEvent.aim_good and cy > basket_y + 10 \
                #         and abs(cx - basket_x) < k_margin_x:
                #     checkEvent.confident += 0.2

            elif conf > 0.5 and classes[int(cls_pred)] == "bib":   #   and (checkEvent.aim_good or checkEvent.event_st):
                checkEvent.confident += 0.2

            elif conf > 0.5 and classes[int(cls_pred)] == "bob" and checkEvent.event_st:
                checkEvent.confident -= 0.6

    else:
        checkEvent.no_bb_cnt += 1
        if checkEvent.no_bb_cnt > 100:
            checkEvent.no_bb_cnt = 100

        checkEvent.confident -= 0.1

    # Next set f_result based on confidence
    if checkEvent.confident >= k_min_score:
        f_result = 2
        checkEvent.event_st = False
        checkEvent.confident = 0
        checkEvent.aim_good = False
        checkEvent.last_ball_posn_y = 100
        if not f_save_video:
            winsound.Beep(1000, 1500)

    elif checkEvent.event_st:
        f_result = 1
        if checkEvent.last_ball_posn_y < 40:  # indicate it is going up, so need to wait for longer time
            if checkEvent.no_bb_cnt > 20:
                checkEvent.event_st = False
                checkEvent.confident = 0
                checkEvent.aim_good = False
                checkEvent.last_ball_posn_y = 100
                f_result = 3
                if not f_save_video:
                    winsound.Beep(1000, 500)
        else:
            if checkEvent.no_bb_cnt > 3:
                checkEvent.event_st = False
                checkEvent.confident = 0
                checkEvent.aim_good = False
                checkEvent.last_ball_posn_y = 100
                f_result = 3
                if not f_save_video:
                    winsound.Beep(1000, 500)

    else:
        f_result = 0
        checkEvent.confident = 0
        checkEvent.aim_good = False
        checkEvent.last_ball_posn_y = 100

    result_enum = ["no_shooting", "shooting_start", "PASS", "FAIL"]
    return result_enum[f_result]


def count_pass_fail(curr_result):
    if not hasattr(count_pass_fail, "total"):
        count_pass_fail.total = 0
        count_pass_fail.pass_num = 0
        count_pass_fail.prev_result = "no_shooting"

    if count_pass_fail.prev_result == "no_shooting" and curr_result == "shooting_start":
        count_pass_fail.total += 1
    elif count_pass_fail.prev_result == "shooting_start" and curr_result == "PASS":
        count_pass_fail.pass_num += 1

    count_pass_fail.prev_result = curr_result
    date_time_limit = datetime(2025, 3, 26, 17, 0)
    # Check if current time is before the limit
    if datetime.now() < date_time_limit:
        return count_pass_fail.pass_num, count_pass_fail.total
    else:
        return 0, 0


def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections.numpy()


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3-tiny.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-c", "--classes", type=str, default="data/coco.names", help="Path to classes label file (.names)")
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("-s", "--f_save_imgs", type=bool, default=False, help="Enable or Disable image saving")
    parser.add_argument("--img_size", type=int, default=224, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names
    model = load_model(args.model, args.weights)
    basketball_realtime(model, args.img_size, args.output, args.conf_thres, args.nms_thres, args.f_save_imgs)


if __name__ == '__main__':

    run()

