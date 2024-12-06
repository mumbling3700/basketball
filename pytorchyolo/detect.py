#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import shutil
import random
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import matplotlib
matplotlib.use('Agg')
import gc
from basketball_realtime import checkEvent, count_pass_fail


def save_yolo_format(file_name, boxes, width, height):
    with open(file_name, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2, conf, cls_pred = box
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w = abs(x2 - x1) / width
            h = abs(y2 - y1) / height
            f.write(f"{int(cls_pred)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def detect_seq_img(model, img_path, img_size, classes, output_path, conf_thresh, nms_thresh):
    """process single images in sequence. then plot and save result in image."""
    png_files = [file for file in os.listdir(img_path) if file.endswith('.png')]
    txt_folder = os.path.join(img_path, 'label')  # this is the folder that keeps auto label
    f_save_processed = True

    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    for i in range(len(png_files)):
        img_full = os.path.join(img_path, png_files[i])
        print(png_files[i])

        image_bgr = cv2.imread(img_full)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # starttime = time.time()
        dets = detect_image(model, image, img_size, conf_thresh, nms_thresh)
        # endtime = time.time()
        # print(f"Execution time: {endtime - starttime} seconds")
        f_result = checkEvent(dets)
        passnum, totalnum = count_pass_fail(f_result)

        # next generate auto label. Need to generate label for all the images in case there is false neg.

        txt_label = os.path.splitext(png_files[i])[0] + ".txt"
        txt_full = os.path.join(txt_folder, txt_label)
        # shutil.copy(img_full, images_folder)
        img_height = image.shape[0]
        img_width = image.shape[1]
        save_yolo_format(txt_full, dets, img_width, img_height)
        if f_save_processed:
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(image)
            # Rescale boxes to original image
            # detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = np.unique(dets[:, -1])
            n_cls_preds = len(unique_labels)
            # Bounding-box colors
            cmap = plt.get_cmap("tab20b")
            colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_pred in dets:
                print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=f"{classes[int(cls_pred)]}: {conf:.2f}",
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0})

            plt.text(
                120,
                80,
                s=f"{passnum} / {totalnum}",
                fontsize=20,
                color='green'
            )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            output = os.path.join(output_path, f"{png_files[i]}")
            plt.savefig(output, bbox_inches="tight", pad_inches=0.0)
            plt.close('all')
            del image  # Delete image object
            gc.collect()  # Force garbage collection to free memory


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
    parser.add_argument("-i", "--images", type=str, default="data/samples", help="Path to directory with images to inference")
    parser.add_argument("-c", "--classes", type=str, default="data/coco.names", help="Path to classes label file (.names)")
    parser.add_argument("-o", "--output", type=str, default="C:/basketball_shooting/c920/output", help="Path to output directory")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=224, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names
    model = load_model(args.model, args.weights)

    try:
        os.makedirs(args.output)
        print(f"Nested directories '{args.output}' created successfully.")
    except FileExistsError:
        print(f"One or more directories in '{args.output}' already exist.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{args.output}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    detect_seq_img(model, args.images, args.img_size,
                   classes, args.output, args.conf_thres, args.nms_thres)





if __name__ == '__main__':
    run()
