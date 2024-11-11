import argparse
import os
from PIL import Image
import concurrent.futures

import config
from utils.contours import crop_to_page, final_crop
from utils.rotation import get_text_orientation

def process_image(image, filename):
  """Process the image by cropping and rotating."""
  # Step 1: Pre-crop the image around the likely text area
  pre_cropped_image = crop_to_page(image, filename)
  cropped_image = final_crop(pre_cropped_image, filename)
  # Step 2: Detect text orientation in the cropped image
  rotation_angle = get_text_orientation(cropped_image)
  # Step 3: Rotate the cropped image based on detected angle
  if rotation_angle != 0:
    rotated_image = cropped_image.rotate(-rotation_angle, expand=True)
  else:
    rotated_image = cropped_image
  # Step 4: Resize the image to reduce its resolution
  width_percent = (config.IMAGE_WIDTH / float(rotated_image.size[0]))
  image_height = int((float(rotated_image.size[1]) * float(width_percent)))
  resized_image = rotated_image.resize((config.IMAGE_WIDTH, image_height), Image.LANCZOS)

  if config.DEBUG:
    output_path = os.path.join(config.OUTPUT_PATH, filename)
    # add .jpg extension if not present
    if not output_path.endswith('.jpg'):
      output_path += '.jpg'
    resized_image.save(output_path)

  print(f'Processing complete: {filename}')
  return resized_image


def images_to_pdf(image_list, input_path):
  """Convert a list of Pillow images to a PDF document."""
  # Ensure there are images to convert
  if not image_list:
    raise ValueError("Image list is empty")
  filename = input_path + ".pdf"
  # Convert all images to RGB mode (required for PDF)
  rgb_images = [img.convert("RGB") for img in image_list]
  # Save the first image and append the rest
  rgb_images[0].save(filename, save_all=True, append_images=rgb_images[1:], format="PDF", dpi=config.DPI)
  print(f"PDF created successfully at {filename}")


# Asynchronous image processing function
def async_process_image(args):
  image_path, filename = args
  img = Image.open(image_path)
  return process_image(img, filename), filename


def main(input_path):
  # Create the output folder if it doesn't exist
  if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)
  else:
    # Clear existing files in the output folder
    for filename in os.listdir(config.OUTPUT_PATH):
      file_path = os.path.join(config.OUTPUT_PATH, filename)
      os.remove(file_path)
  if os.path.exists(f"{config.OUTPUT_PATH}.pdf"):
    os.remove(f"{config.OUTPUT_PATH}.pdf")

  # Process images asynchronously using ThreadPoolExecutor
  images = []
  tasks = []
  input_path = input_path.rstrip('/')
  root, ext = os.path.splitext(input_path)
  save_path = root.split('/')[-1]
  # If inputfile_path ends with .jpg, process a single image
  if ext == ".jpg":
    tasks.append((input_path, save_path))
  else:
    for i, filename in enumerate(sorted(os.listdir(input_path))):
      if config.DEBUG and i >= 40:  # limit files in debug mode
        break
      if filename.endswith(".jpg"):
        image_path = os.path.join(input_path, filename)
        tasks.append((image_path, filename[:-4]))

  print(f"Processing {len(tasks)} images ...")
  # Use ThreadPoolExecutor for asynchronous processing
  with concurrent.futures.ProcessPoolExecutor() as executor:
    future_to_image = {executor.submit(async_process_image, task): task for task in tasks}
    for future in concurrent.futures.as_completed(future_to_image):
      try:
        processed_image, filename = future.result()
        images.append((processed_image, filename))
      except Exception as exc:
        if config.DEBUG:
          print(f"Error processing image {future_to_image[future][1]}: {exc}")

  # Sort images by filename to preserve the correct order
  images.sort(key=lambda x: x[1])

  # Convert processed images (only the images, not the filenames) to a PDF
  images_to_pdf([img[0] for img in images], f"output/{save_path}")
  print("All images processed and PDF created.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Process images from input folder and save to output folder.")
  parser.add_argument("input_path", type=str, help="Path to the input folder containing images")
  parser.add_argument("--threshold", type=int, help="Threshold value for detecting book folds (optional)")
  parser.add_argument("--debug", action='store_true', help="Enable debug mode (optional)")
  args = parser.parse_args()
  config.DEBUG = args.debug
  if args.threshold is not None:
    config.FOLD_THRESHOLD = args.threshold
  main(input_path=args.input_path)
