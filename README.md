# TWARDISv0.1

All scripts used to generate the results presented in manuscript.

For image analysis, the Meta SAM model sam_vit_h_4b8939 was used.
For video analysis, the Meta SAM2 repository was cloned and the model downloaded from the July 28th 2024 version (https://github.com/facebookresearch/sam2/).
Our fine-tuned worm classifier is available on HuggingFace (https://huggingface.co/lillyguisnet/celegans-classifier-vit-h-14-finetuned).


## Python Dependencies
Throughout this repo, we will be making use of a number of Python libraries. Here is a list of those libraries with a general description of how these libraries are generally used.

- `cv2`: This is a general purpose computer vision library used by many professionals across many industries for understand and working with images or videos.
- `numpy`: This is a popular Python library largely used by data science and data analytics professionals. It offers many different functions for working with large numerical data.
- `re`: "Re" is short for "Regular Expressions", or as it is more commonly known, Regex. Regex is a standardized way to search for certain character patterns in a string.
- `tifffile`: This is used to generally work with TIFF files in Python. Because of the subject matter in question here, it is natural to expect that a user would be storing their files as TIFF files.


## About the Code
In this section, we'll break down what each respective directory does with a plain English explanation. (Note: I am helping a friend with this project given my machine learning experience, but I am not at all familiar the direct subject matter. Please forgive my ignorance if I say anything goofy. 🙏🏽)

### RIA Calcium Imaging (`/RIA_calcium_imaging`)

#### 1. Preparing the files to work with Meta's SAM model. (`1_tiftojpg.py`)
**General description**: This file processes through video or image files and prepares them for use with Meta's SAM model. This file expects that you already have these images / videos downloaded to your local machine. There is no offering for any sample data. This script generally expects that you are working with TIFF files, but it also does support a limited number of file types. The processed files will be output as JPG files.

**Python dependencies**: The following Python libraries are used to support this script:

- `cv2`: In this script, `cv2` is being leveraged to work with video files, get attribute data about images / videos (e.g. frame width, frame height, etc.), and writing out the prepared images that we will use with Meta's SAM model.
- `numpy`: In this specific file, `numpy` is being minimally used, specifically being used to get the minimum and maxiumum frames as well as being used to set an integer data type.
- `re`: In this specific file, Regex is being minimally used, only to extract the numeric frame index from filenames.
- `tifffile`: This is being used to interact with input TIFF files.