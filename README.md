# ACE Images

Scripts that are useful for processing ACE document scans:

* Crop and rotate images and create a PDF
* Extract text from the PDF using OCR

These scans are assumed to have the fold of the book near the bottom of the image and for the images to be in .jpg format.

## Installation

This script uses Tesseract to determine the rotation of a page. To install Tesseract, run this command in Ubuntu: `sudo apt-get install tessaract-ocr`. If you're using another operating system, please refer to the [Tesseract documentation](https://github.com/tesseract-ocr/tesseract/wiki) for installation instructions.

Pull the repo and setup a python virtual environment. Then install the dependencies.

```bash
git clone git@gitlab.scholarsportal.info:ai-ml/ace-images.git
cd ace-images
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Scans to PDF

Crops and rotates raw book scans and saves images as pages in a pdf file.

```bash
python3 scans_to_pdf.py path_to_image_directory_or_single_image --threshold 50 --debug
```

The debug flag will limit processing to 40 pages and also create additional images showing the intermediate stages of detecting fold lines and cropping. The threshold is used for determining if an edge is a fold line or not. Use higher values if some folds are not detected and lower values if pages are getting cut off.

Output will be saved in the *output* directory.

## OCR

You can use the `ocr_pdf.py` script to perform OCR on a pdf file. This script uses Tesseract to detect text in a pdf and then saves it to a text file.

```bash
python3 ocr_pdf.py path_to_pdf_file --dpi 150 --contrast 0.75 --lang eng --debug
```

The *debug* flag will limit processing to 30 pages. The *dpi* setting controls the resolution of the images used for OCR. Try to match the input document resolution. The *contrast* setting is a multiplier that can be used to adjust the contrast of the image before OCR. Use higher values if some text is not detected and lower values if too much text is detected. If you are getting lots of garbage text, try lowering the contrast to 0.8 or even 0.7. The *lang* setting controls the language used for OCR. You can use multiple languages separated by a + sign, e.g., `--lang eng+fra` (default). The output will be saved as a text file in the same folder as the input pdf.
