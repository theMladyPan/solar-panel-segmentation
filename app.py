#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import gradio as gr
import logging
from typing import Any
import tempfile
import zipfile
import os

log = logging.getLogger("werkzeug")
log.setLevel(logging.INFO)


valid_contours: list[Any]


# Function for Step 1: Blurring
def blur_image(image_path, blurks):
    gray = cv2.imread(image_path)
    if gray is None:
        return "Invalid image path!"

    # Convert the image to grayscale
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Invert the image
    gray = cv2.bitwise_not(gray)

    blurred = cv2.GaussianBlur(gray, (blurks, blurks), 0)
    return blurred


# Function for Step 2: Adaptive Thresholding
def adaptive_threshold(blurred_image, atks, atoff):
    # if src.type() == CV_8UC1
    if blurred_image.ndim == 3:
        blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    # Apply Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, atks, atoff)
    return thresh


# Function for Step 3: Contour Detection
def find_contours(original_image, thresh_image, min_area, max_area, aspect_ratio_min, aspect_ratio_max, ksclose):
    global valid_contours
    valid_contours = []
    # convert closed image to grayscale
    if thresh_image.ndim == 3:
        thresh_image = cv2.cvtColor(thresh_image, cv2.COLOR_BGR2GRAY)

    inverted = cv2.bitwise_not(thresh_image)

    # Perform morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksclose, ksclose))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)

    # invert edges
    inv_edges = cv2.bitwise_not(closed)
    contours, _ = cv2.findContours(inv_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = original_image.copy()
    # convert to RGB if grayscale
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    # offset = 10
    for contour in contours:
        # Filter based on area to ignore noise
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            # draw bounding box in red
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            log.info(f"Aspect Ratio: {aspect_ratio}")
            if aspect_ratio > aspect_ratio_min and aspect_ratio < aspect_ratio_max:
                # y0, y1 = max(0, y - offset), min(y + h + offset, output.shape[0])
                # x0, x1 = max(0, x - offset), min(x + w + offset, output.shape[1])
                # cv2.rectangle(output, (x0, y0), (x1, y1), (0, 0, 255), 1)

                # draw polygon too
                epsilon = 0.1 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(output, [approx], 0, (0, 255, 0), 2)
                valid_contours.append(contour)

    return [closed, output, len(valid_contours)]


def download_images(image_path):
    global valid_contours
    original_image = cv2.imread(image_path)
    original_image_width = original_image.shape[1]
    original_image_height = original_image.shape[0]

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "cropped.zip")

    offset = 10
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for i, contour in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(contour)
            y0, y1 = max(0, y - offset), min(y + h + offset, original_image_height)
            x0, x1 = max(0, x - offset), min(x + w + offset, original_image_width)

            roi = original_image[y0:y1, x0:x1]

            # Create a temporary file for the ROI
            temp_file_path = os.path.join(temp_dir, f"roi_{i}.png")
            cv2.imwrite(temp_file_path, roi)

            # Add the file to the ZIP archive
            zipf.write(temp_file_path, f"roi_{i}.png")

    # return zip_path
    return gr.DownloadButton(label=f"Download {zip_path}", value=zip_path, visible=True)


def all_steps(image_path, blurks, atks, atoff, min_area, max_area, aspect_ratio_min, aspect_ratio_max, ksclose):
    blurred = blur_image(image_path, blurks)
    thresh = adaptive_threshold(blurred, atks, atoff)
    closed, output, n_segments = find_contours(
        blurred, thresh, min_area, max_area, aspect_ratio_min, aspect_ratio_max, ksclose
    )
    return [blurred, thresh, closed, output, n_segments]


def process_steps(blurred, atks, atoff, min_area, max_area, aspect_ratio_min, aspect_ratio_max, ksclose):
    thresh = adaptive_threshold(blurred, atks, atoff)
    closed, output, n_segments = find_contours(
        blurred, thresh, min_area, max_area, aspect_ratio_min, aspect_ratio_max, ksclose
    )
    return [blurred, thresh, closed, output, n_segments]


# Gradio App Using Blocks
with gr.Blocks() as app:
    gr.Markdown("# Multi-Step Image Segmentation App")
    gr.Markdown("Each step has its own sliders and output. Adjust sliders to control each step.")

    # Step 1: Blurring
    with gr.Row():
        with gr.Column(scale=1):
            blurks_slider = gr.Slider(1, 25, step=2, value=7, label="Blur Kernel Size (blurks)")
            image_path = gr.Textbox(label="Image Path", value="img/img7c.JPG")
            # Button Actions
            load_button = gr.Button("Load Image")
            atks_slider = gr.Slider(3, 25, step=2, value=15, label="Adaptive Threshold Kernel Size (atks)")
            atoff_slider = gr.Slider(-20, 20, step=1, value=5, label="Adaptive Threshold Offset (atoff)")
        with gr.Column(scale=1):
            blurred_output = gr.Image(label="Blurred Image")
        with gr.Column(scale=1):
            threshold_output = gr.Image(label="Adaptive threshold")

    # Step 3: Contour Detection
    with gr.Row():
        with gr.Column(scale=1):
            ksclose_slider = gr.Slider(1, 20, step=1, value=13, label="Kernel Size for Closing (ksclose)")
            min_area_slider = gr.Slider(100, 2000, step=20, value=440, label="Min Contour Area (min_area)")
            max_area_slider = gr.Slider(100, 5000, step=50, value=1000, label="Max Contour Area (max_area)")
            aspect_ratio_min_slider = gr.Slider(
                0.1, 5.0, step=0.1, value=0.3, label="Min Aspect Ratio (aspect_ratio_min)"
            )
            aspect_ratio_max_slider = gr.Slider(
                0.1, 5.0, step=0.1, value=0.6, label="Max Aspect Ratio (aspect_ratio_max)"
            )
        with gr.Column(scale=1):
            mask_image = gr.Image(label="Contour Mask")
        with gr.Column(scale=1):
            contour_output = gr.Image(label="Segments")
    with gr.Row():
        with gr.Column(scale=1):
            n_segments_text = gr.Textbox(label="Number of Segments", value="0")
        with gr.Column(scale=1):
            prepare_button = gr.Button("Prepare Images")
        with gr.Column(scale=1):
            download_button = gr.DownloadButton(visible=False)

    all_steps_args = {
        "fn": all_steps,
        "inputs": [
            image_path,
            blurks_slider,
            atks_slider,
            atoff_slider,
            min_area_slider,
            max_area_slider,
            aspect_ratio_min_slider,
            aspect_ratio_max_slider,
            ksclose_slider,
        ],
        "outputs": [blurred_output, threshold_output, mask_image, contour_output, n_segments_text],
    }

    process_steps_args = {
        "fn": process_steps,
        "inputs": [
            blurred_output,
            atks_slider,
            atoff_slider,
            min_area_slider,
            max_area_slider,
            aspect_ratio_min_slider,
            aspect_ratio_max_slider,
            ksclose_slider,
        ],
        "outputs": [blurred_output, threshold_output, mask_image, contour_output, n_segments_text],
    }

    # Bind Functions to UI
    load_button.click(**all_steps_args)
    prepare_button.click(
        fn=download_images,
        inputs=[image_path],
        outputs=[download_button],
    )

    # Bind Sliders to Functions
    blurks_slider.change(**all_steps_args)
    atks_slider.change(**process_steps_args)
    atoff_slider.change(**process_steps_args)
    ksclose_slider.change(**process_steps_args)

    min_area_slider.change(**process_steps_args)
    max_area_slider.change(**process_steps_args)
    aspect_ratio_min_slider.change(**process_steps_args)
    aspect_ratio_max_slider.change(**process_steps_args)

# Launch the App
app.launch()
