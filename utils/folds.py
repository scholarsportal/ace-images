import cv2
import numpy as np

import config


def merge_lines(lines, y_tolerance=10):
  """Merge lines that are close to each other on the y-axis."""
  if lines is None:
    return []
  merged_lines = []
  for line in lines:
    x1, y1, x2, y2 = line[0]
    # Ensure that the line points are sorted
    y_top, y_bottom = sorted([y1, y2])
    x_left, x_right = sorted([x1, x2])
    # if the line is not almost horizontal, throw it out
    if (x_right - x_left) / (y_bottom - y_top + 1) < 50:
      continue
    added = False
    for merged_line in merged_lines:
      m_left, m_top, m_right, m_bottom = merged_line
      # Check if y-values are within the tolerance
      if abs((y_top + y_bottom - m_top - m_bottom) / 2) < y_tolerance:
        # Merge the lines by extending the endpoints
        merged_line[0] = min(m_left, x_left)
        merged_line[1] = min(m_top, y_top)
        merged_line[2] = max(m_right, x_right)
        merged_line[3] = max(m_bottom, y_bottom)
        added = True
        break
    if not added:
      # If the line isn't close to any existing ones, add it as a new merged line
      merged_lines.append([x_left, y_top, x_right, y_bottom])
  # Sort merged lines by their vertical position (average of y1 and y2)
  merged_lines.sort(key=lambda line: (line[1] + line[3]))
  return merged_lines


def detect_fold(contour, image_np):
  """Detect the fold in the bottom 1/4 of the contour and modify the contour if found."""
  x, y, w, h = cv2.boundingRect(contour)
  # blurred = cv2.medianBlur(image_np, 1)
  contour_area = image_np[y:y + h, x:x + w]
  # Focus on the bottom quarter of the contour area
  bottom_y = int(h * 3 / 4)
  bottom_part = contour_area[bottom_y:, :]
  # Apply Canny edge detection to the bottom third
  edges = cv2.Canny(bottom_part, 10, 90)
  # Use Hough Line Transform to detect lines
  lines = cv2.HoughLinesP(
      edges,
      1,
      np.pi / 180,
      threshold=config.FOLD_THRESHOLD,
      # minLineLength=int(w * 0.7),
      maxLineGap=3)
  merged_lines = merge_lines(lines, y_tolerance=30)
  fold_y = 0
  if len(merged_lines) > 0:
    filtered_lines = []
    if config.DEBUG:
      print(f"Fold lines detected. Image Width: {w}, Height: {h}, Veritical position: {y}")
    for line in merged_lines:
      x1, y1, x2, y2 = line
      if config.DEBUG:
        print(f"x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")
      if (x2 - x1) > w * 0.75:  # almost spans the width
        if y1 > (0.1 *
                 (h - bottom_y)) and y1 < (0.8 *
                                           (h - bottom_y)):  # not near the edges of the bottom area
          # add y1 and width to filered_lines
          filtered_lines.append(((y1 + y2) / 2, x2 - x1))
          if config.DEBUG:
            print("  line accepted")
    if len(filtered_lines) > 0:
      # sort filtered_lines by x2 - x1, and then by y1
      filtered_lines.sort(key=lambda x: (x[1], x[0]), reverse=True)
      # sorted_filtered_lines = sorted(filtered_lines, key=lambda x: x[1], reverse=True)
      fold_y = filtered_lines[0][0] + bottom_y  # Adjust fold_y to full contour height

  if fold_y != 0:
    if config.DEBUG:
      print(f"fold_y: {fold_y}, Adjustment: {h + y - fold_y}")
    # Create a rectangular contour from the bounding rectangle
    rect_contour = np.array(
        [
            [[x, y]],  # Top-left corner
            [[x + w, y]],  # Top-right corner
            [[x + w, y + fold_y]],  # Bottom-right corner, adjusted
            [[x, y + fold_y]]  # Bottom-left corner, adjusted
        ],
        dtype=np.int32)
    return rect_contour
  else:
    # No fold detected, return the original contour
    return contour
