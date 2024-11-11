import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps, ImageEnhance

import config


def crop_image(image, size):
  width, height = image.size
  new_width = int(width * size)
  new_height = int(height * size)
  start_x = (width - new_width) // 2
  start_y = (height - new_height) // 2
  cropped_image = image.crop((start_x, start_y, start_x + new_width, start_y + new_height))
  return cropped_image


def fine_grained_rotation(image):
  """Detect fine-grained rotation using edge detection and Hough transform with OpenCV."""
  image_np = np.array(image)
  blurred = cv2.medianBlur(image_np, 1)
  edges = cv2.Canny(blurred, 25, 70)
  lines = cv2.HoughLines(edges, 0.9, np.pi / 180, 300)
  if lines is None:
    return 0
  angles = []
  for line in lines:
    for rho, theta in line:
      angle = np.rad2deg(theta)
      if 85 < angle < 95:  # Filter near-vertical lines
        # if DEBUG:
        #   print(f"hough line angle: {angle}")
        angles.append(angle)
  # Compute the median angle to avoid outliers
  if angles:
    median_angle = sum(angles) / len(angles) - 90  # Adjust back
  else:
    median_angle = 0
  if config.DEBUG:
    print(f"Fine rotation: {-median_angle} degrees")
  return -median_angle


def get_box_orientation(image):
  """Determine the text orientation of a part of the page."""
  grayscale_image = ImageOps.grayscale(image)
  grayscale_image.info['dpi'] = (100, 100)
  rotation_angle = 0
  # Use pytesseract to detect orientation
  try:
    osd = pytesseract.image_to_osd(grayscale_image, output_type=pytesseract.Output.DICT)
    rotation_confidence = osd['orientation_conf']
    # Throw if the confidence is lower than 2.0 and try alternative approach
    if rotation_confidence < 2.0:
      raise ValueError("Rotation confidence too low.")
    rotation_angle = osd['rotate']
  except Exception as e:
    try:
      # alternative image processing
      image_np = np.array(grayscale_image)
      # blur and enhance
      blurred = cv2.medianBlur(image_np, 1)  # get rid of some noise
      # enhanced_image = cv2.equalizeHist(blurred)
      processed_image = Image.fromarray(blurred)
      processed_image.info['dpi'] = (200, 200)
      osd = pytesseract.image_to_osd(processed_image, output_type=pytesseract.Output.DICT)
      rotation_confidence = osd['orientation_conf']
      # Throw if the confidence is too low
      if rotation_confidence < 2.0:
        raise ValueError("Rotation confidence too low.")
      rotation_angle = osd['rotate']
      if config.DEBUG:
        print("*** Using alternative orientation detection ***")
    except Exception as ex:
      raise ex # re-raise exception
  return rotation_angle

def get_text_orientation(image):
  """Determine the text orientation using Tesseract OCR."""
  # crop image into a left and right half
  width, height = image.size
  middle = width // 2
  left_half = image.crop((0, 0, middle, height))
  right_half = image.crop((middle, 0, width, height))
  cropped_image = crop_image(image, 0.95)
  rotation_angle = 0.0
  try:
    rotation_angle = get_box_orientation(cropped_image)
  except:
    try:
      if config.DEBUG:
        print("*** Trying LEFT half for rotation detection ***")
      rotation_angle = get_box_orientation(left_half)
    except:
      try:
        if config.DEBUG:
          print("*** Trying RIGHT half for rotation detection ***")
        rotation_angle = get_box_orientation(right_half)
      except Exception as ex:
        if config.DEBUG:
          print(f"Error in get_text_orientation: {str(ex)}")
  # Check if the image is landscape or portrait
  if rotation_angle == 0 and width > height:
    rotation_angle = 90  # Rotate by 90 degrees to make it portrait
  # Rotate the image based on the detected coarse rotation angle
  rotated_image = cropped_image.rotate(-rotation_angle, expand=True)
  # Step 2: Use OpenCV to detect fine-grained rotation adjustment
  fine_grained_angle = fine_grained_rotation(rotated_image)
  # Total rotation is the sum of coarse and fine-grained adjustments
  total_rotation_angle = rotation_angle + fine_grained_angle

  return total_rotation_angle
