# TWARDISv0.1

All scripts used to generate the results presented in manuscript.

For image analysis, the Meta SAM model sam_vit_h_4b8939 was used.
For video analysis, the Meta SAM2 repository was cloned and the model downloaded from the July 28th 2024 version (https://github.com/facebookresearch/sam2/).
Our fine-tuned worm classifier is available on HuggingFace (https://huggingface.co/lillyguisnet/celegans-classifier-vit-h-14-finetuned).


## Quick Note: About Me
Hi all. 👋🏽 I am a professional machine learning engineer, and I have a friend currently going through a Masters program that was wanting a deeper understanding of how to use this code. Because I am familiar with Python and machine learning, I am very happy to help; however, I am admittedly very ignorant about the subject matter at hand. I'm going to do my best to try explaining this repo, but please forgive me if my dummy self says something stupid about the non-ML-related / non-Python-related subject matter.

## Python Dependencies
Throughout this repo, we will be making use of a number of Python libraries. Here is a list of those libraries with a general description of how these libraries are generally used.

- `cv2`: This is a general purpose computer vision library used by many professionals across many industries for understand and working with images or videos.
- `multiprocessing`: At the heart of your computer is a CPU (or maybe GPU) which does all the computational processing on your computer. Your computer actually likely has multiple cores that can run multiple computational processes in parallel, in what we would refer to as a **thread**. Generally speaking, Python defaults to running things in a single thread, which is not ideal if we expect a large computational load. (Such as processing a bunch of TIFFs into JPGs.) As such, we can maximize what our system is doing by introducing multiprocessing, which runs computational processes in multiple threads. For example, if you had 4 cores in your CPU, you may theoretically be able to process 4 TIFFs into their respective 4 sets of output images in parallel fashion.
- `numpy`: This is a popular Python library largely used by data science and data analytics professionals. It offers many different functions for working with large numerical data.
- `re`: "Re" is short for "Regular Expressions", or as it is more commonly known, Regex. Regex is a standardized way to search for certain character patterns in a string.
- `tifffile`: This is used to generally work with TIFF files in Python. Because of the subject matter in question here, it is natural to expect that a user would be storing their files as TIFF files.
- `tqdm`: This is short for the Arabic word "taqaddum", which means "progress". Sounds fancy, but all we use `tqdm` for is displaying progress bars. 😁


## About the Code
In this section, we'll break down what each respective directory does with a plain English explanation.

### RIA Calcium Imaging (`/RIA_calcium_imaging`)

#### 1. Preparing the files to work with Meta's SAM model. (`1_tiftojpg.py`)
**General description**: This file processes through video or image files and prepares them for use with Meta's SAM model. This file expects that you already have these images / videos downloaded to your local machine. There is no offering for any sample data. This script generally expects that you are working with TIFF files, but it also does support a limited number of file types. The processed files will be output as JPG files.

**Python dependencies**: The following Python libraries are used to support this script:

- `cv2`: This is being leveraged to work with video files, get attribute data about images / videos (e.g. frame width, frame height, etc.), and writing out the prepared images that we will use with Meta's SAM model.
- `multiprocession`: This is being used to process the TIFFs into output JPGs using parallel processing on your local CPU.
- `numpy`: This is being minimally used, specifically being used to get the minimum and maxiumum frames as well as being used to set an integer data type.
- `re`: Regex is being minimally used, only to extract the numeric frame index from filenames.
- `tifffile`: This is being used to interact with input TIFF files.
- `tqdm`: This is used to display a progress bar as the files are processing.