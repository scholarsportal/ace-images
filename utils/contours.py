import cv2
import numpy as np
from PIL import Image

import config
from utils.folds import detect_fold

def contour_to_rectangle(contour):
  x, y, w, h = cv2.boundingRect(contour)
  rect_contour = np.array(
      [
          [[x, y]],  # Top-left corner
          [[x + w, y]],  # Top-right corner
          [[x + w, y + h]],  # Bottom-right corner
          [[x, y + h]]  # Bottom-left corner
      ],
      dtype=np.int32)
  return rect_contour


def filter_contours(contours, image_np):
  """Filter out contours that are too small, not page-like, and then sort by solidity."""
  height, width = image_np.shape[:2]
  valid_contours = []
  for contour in contours:
    # Get bounding box and area of the contour
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    aspect_ratio = max(w / h, h / w)
    # Calculate contour area and convex hull area (to get solidity)
    contour_area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    # Ensure convex hull area is not zero (to avoid division by zero)
    if hull_area == 0:
      continue
    solidity = float(contour_area) / hull_area  # Ratio of contour area to convex hull area
    # Filter based on area, aspect ratio, and solidity threshold
    if (0.1 * height * width < area < 0.9 * height * width) and (aspect_ratio < 1.7):
      valid_contours.append((contour, solidity))

  # Sort valid contours by solidity in descending order (highest solidity first)
  if valid_contours:
    valid_contours.sort(key=lambda x: x[1], reverse=True)

  # Return only the sorted contours, discarding the solidity values
  return [contour for contour, solidity in valid_contours]


def final_crop(image, filename):
  """Remove any remaining bars around the page."""
  # Convert the image to a NumPy array for OpenCV
  image_np = np.array(image)
  gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
  blurred = cv2.medianBlur(gray, 71)
  _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
  # Perform morphological closing to remove small gaps and spikes
  kernel = np.ones((200, 200), np.uint8)
  closed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
  contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  x, y, w, h = 9999, 9999, 0, 0
  if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    # Approximate the contour to a polygon (force a rectangle shape)
    # epsilon = 0.2 * cv2.arcLength(largest_contour, True)
    # approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    x, y, w, h = cv2.boundingRect(largest_contour)
  else:
    return image
  # Expand contour if other contours exist
  max_x, max_y = 0, 0
  for c in contours:
    xc, yc, wc, hc = cv2.boundingRect(c)
    if xc < x: x = xc
    if yc < y: y = yc
    if xc + wc > max_x: max_x = xc + wc
    if yc + hc > max_y: max_y = yc + hc
  w = max_x - x
  h = max_y - y

  cropped_image_np = image_np[y:y + h, x:x + w]
  cropped_image = Image.fromarray(cropped_image_np)
  if config.DEBUG:
    debug_image = image_np.copy()
    # Draw all contours found on the debug image
    cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 3)
    cv2.drawContours(debug_image, [largest_contour], -1, (255, 255, 0), 3)
    cv2.imwrite(f"./output/debug_final_{filename}.jpg", debug_image)

  return cropped_image


def crop_to_page(image, filename):
  """Automatically crop the image to the page's text area using OpenCV."""
  # Convert the image to a NumPy array for OpenCV
  image_np = np.array(image)
  gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
  contours = None
  # Highlight borders
  if filename == "0000" or filename == "0001":
    print(f"Processing Book Cover ...")
    blurred = cv2.medianBlur(gray, 71)
    _, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    # Perform morphological closing to remove small gaps and spikes
    kernel = np.ones((200, 200), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  else:
    blurred = cv2.medianBlur(gray, 31)
    # adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    #                                41, 1)
    _, binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
    # Perform morphological closing to remove small gaps and spikes
    kernel = np.ones((200, 200), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Find contours which should represent text areas
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  filtered_contours = filter_contours(contours, image_np)
  if not filtered_contours:
    # try alternative preparation to get boundaries
    blurred = cv2.medianBlur(gray, 41)
    # blurred = cv2.GaussianBlur(gray, (61, 61), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                   71, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filter_contours(contours, image_np)
    if config.DEBUG and filtered_contours:
      print("No inverted contours found. Using alternative blur method")

  if filtered_contours:
    # Get bounding box of the smallest contour
    # page_contour = min(filtered_contours, key=cv2.contourArea)

    # Get the contour with the highest solidity (first in the list)
    page_contour = filtered_contours[0]
    rect_contour = contour_to_rectangle(page_contour)
    # Folds aren't detected nicely sometimes, remove it
    modified_contour = detect_fold(rect_contour, gray)

    x, y, w, h = cv2.boundingRect(modified_contour)
    if config.DEBUG:
      print(f"De-folded box: x={x}, y={y}, w={w}, h={h}")
      debug_image = image_np.copy()
      # Draw all contours found on the debug image
      cv2.drawContours(debug_image, filtered_contours, -1, (0, 255, 0), 3)
      cv2.drawContours(debug_image, [rect_contour], -1, (0, 0, 255), 3)
      cv2.drawContours(debug_image, [modified_contour], -1, (255, 255, 0), 3)
      cv2.imwrite(f"./output/debug_{filename}.jpg", debug_image)

    # Crop the image to the bounding box
    cropped_image_np = image_np[y:y + h, x:x + w]
    # Convert back to PIL Image for consistency
    cropped_image = Image.fromarray(cropped_image_np)
    return cropped_image
  else:
    if config.DEBUG:
      print("No contours found. Returning original image.")
    return image
