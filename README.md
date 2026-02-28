# TWARDISv0.1

All scripts used to generate the results presented in manuscript.

For image analysis, the Meta SAM model sam_vit_h_4b8939 was used.
For video analysis, the Meta SAM2 repository was cloned and the model downloaded from the July 28th 2024 version (https://github.com/facebookresearch/sam2/).
Our fine-tuned worm classifier is available on HuggingFace (https://huggingface.co/lillyguisnet/celegans-classifier-vit-h-14-finetuned).


## Quick Note: About Me (David Hundley)
Hi all. 👋🏽 I am a professional machine learning engineer, and I have a friend currently going through a Masters program that was wanting a deeper understanding of how to use this code. Because I am familiar with Python and machine learning, I am very happy to help; however, I am admittedly very ignorant about the subject matter at hand. I'm going to do my best to try explaining this repo, but please forgive me if my dummy self says something stupid about the non-ML-related / non-Python-related subject matter.

Everything written in this section was added by me. Part of it was written manually by me, but I also used assistance from AI, specifically `gpt-5.3-codex`. I largely used this to help me write the "About the Code" section. Specifically, I manually write the formatting and description for the first file (`/RIA_calcium_imaging/1_tiftojpg.py`) and asked the AI to follow my same styling and formatting to write the rest of the descriptions. I also used it to create the "Python Dependencies" section. 

## Meta's SAM 2
As you can see in the author's original intro, we're working with Meta's SAM 2 model. Additionally, you can see that the author has fine-tuned their own instance of the model for worm classification, which they have uploaded to Hugging Face.

A quick note about the usage of the SAM 2 code: some of this code **requires** that you have already cloned Meta's SAM 2 model, as this code relies on that repo to run some stuff in this repo. You can find the code you need to clone at this link: https://github.com/facebookresearch/sam2.

## Python Dependencies
If you're new to programming, think of Python libraries as "toolboxes" that each do one kind of job. You do **not** need to master all of these. Most of this repo is built from a small core set, and the rest are helper tools.

### 1) AI model + segmentation (the "find the worm/region" tools)
- `torch` (PyTorch): Runs the deep-learning models.
- `sam2`: Meta's Segment Anything 2 code used to create masks from click prompts.
- `torchvision`: Helper package used with PyTorch for model definitions and image transforms.

### 2) Working with images and videos
- `cv2` (OpenCV): Reads videos, writes frame images, resizes/crops, and does basic image operations.
- `PIL` (Pillow): Simple image loading/saving utilities.
- `tifffile`: Reads TIFF microscopy files (including high bit-depth TIFFs).
- `matplotlib`: Makes plots and visual overlays so you can check whether segmentation looks right.

### 3) Scientific math and image analysis
- `numpy`: Core array math (used almost everywhere).
- `scipy`: Signal/image processing helpers (smoothing, interpolation, distance transforms, etc.).
- `skimage` (scikit-image): Morphology/skeleton tools for shape analysis.
- `networkx`: Graph/pathfinding utilities used in advanced skeleton cleanup steps.

### 4) Saving outputs and making tables
- `h5py`: Saves/loads large mask data in `.h5` format.
- `pandas`: Builds and exports tables (usually CSV-like outputs).
- `pickle`: Saves intermediate Python objects between steps.
- `json`: Stores prompt metadata and other lightweight config data.

### 5) Workflow helpers (convenience tools)
- `tqdm`: Progress bars so long-running scripts show activity.
- `multiprocessing`: Uses multiple CPU cores to speed up heavy jobs.
- `glob`, `pathlib`, `re`: File searching, cleaner path handling, and filename pattern matching.

### 6) Python built-ins you'll see often
- `os`, `sys`, `shutil`: File system operations and path setup.
- `time`, `random`: Timing and random selection utilities.
- `copy`, `gc`, `traceback`, `collections`: Object copying, memory cleanup, error tracing, and helper data structures.

### If your friend only learns 6 things first
Start here: `numpy`, `cv2`, `torch`, `sam2`, `pandas`, and `h5py`.

Everything else is useful support, but those six explain most of what the pipeline is doing.


## About the Code
In this section, we'll break down what each respective directory does with a plain English explanation.

### <font color="#0B7285">RIA Calcium Imaging (`/RIA_calcium_imaging`)</font>
At a high level, this folder is one full pipeline for turning raw imaging videos into biologically useful measurements. The flow is: (1) convert source files into SAM-friendly frame images, (2) crop to the RIA region so the model focuses on the right area, (3) run segmentation of the key compartments, (4) extract brightness/orientation values from those segmentations, (5) segment the head/body region for posture analysis, and then (6) compute head-angle dynamics over time. So conceptually, it goes from **raw pixels → clean segmentations → quantitative neuroscience features**.



<br>

#### <font color="#0B7285">1. Preparing the files to work with Meta's SAM model. (`1_tiftojpg.py`)</font>
**General description**: This file processes through video or image files and prepares them for use with Meta's SAM model. This file expects that you already have these images / videos downloaded to your local machine. There is no offering for any sample data. This script generally expects that you are working with TIFF files, but it also does support a limited number of file types. The processed files will be output as JPG files.

**Python dependencies**: The following Python libraries are used to support this script:

- `cv2`: This is being leveraged to work with video files, get attribute data about images / videos (e.g. frame width, frame height, etc.), and writing out the prepared images that we will use with Meta's SAM model.
- `numpy`: This is being minimally used, specifically being used to get the minimum and maxiumum frames as well as being used to set an integer data type.
- `re`: Regex is being minimally used, only to extract the numeric frame index from filenames.
- `tifffile`: This is being used to interact with input TIFF files.
- `tqdm`: This is used to display a progress bar as the files are processing.

**Expected inputs**:
- Source microscopy files (typically TIFF stacks/videos, with limited support for some other image/video formats).
- A local input directory containing the raw files to be converted.

**Expected outputs**:
- JPEG frame/image files written to an output directory for SAM/SAM2-compatible downstream use.
- Frame naming/indexing normalized so later scripts can process in sequence.

**External dependencies**:
- No third-party repository dependency for this step.
- If videos are used, codec support depends on your local OpenCV/FFmpeg build.



<br>

#### <font color="#0B7285">2. Cropping around the RIA region. (`2_crop_RIAregion.py`)</font>
**General description**: This script picks an unprocessed frame-folder, runs SAM2 with a generic RIA click prompt, propagates that mask through the video, and then creates a fixed-size crop around the detected region for every frame. In other words: this is the "zoom in on RIA" prep step before more detailed segmentation.

**Python dependencies**:
- `torch`: Runs SAM2 inference and CUDA setup.
- `sam2` (from cloned Meta repo): Provides the video predictor used for prompt + propagation.
- `numpy`: Handles points, labels, and mask math.
- `cv2`: Reads/writes frames and performs crop/resize operations.
- `PIL` / `matplotlib`: Used for optional prompt/mask visualization outputs.
- `os` / `sys` / `random`: File traversal, path setup for SAM2 import, and random unprocessed-video selection.
- `tqdm`: Progress bar while cropping all frames.

**Expected inputs**:
- Frame folders (typically generated by `1_tiftojpg.py`) for one video/time-series at a time.
- A generic RIA prompt point definition used to initialize tracking/propagation.
- SAM2 model config/checkpoint paths available in your environment.

**Expected outputs**:
- A cropped-frame directory per source sequence, centered on the propagated RIA region.
- Optional prompt/mask visualization images for quick QA.

**External dependencies**:
- Requires Meta SAM2 code from https://github.com/facebookresearch/sam2 (cloned locally/importable).
- Requires SAM2 model weights/checkpoints downloaded separately from this repo.
- GPU/CUDA is strongly recommended for practical runtime.

<br>

#### <font color="#0B7285">3. Autoprompted compartment segmentation. (`3_autoprompted_RIAsegmentation.py`)</font>
**General description**: This is the main RIA segmentation pipeline. It appends reusable prompt frames to each cropped video, injects object-specific click prompts (from a JSON prompt file), runs SAM2 propagation, performs QA/error checks (empty masks, oversized masks, overlaps, distance checks), optionally makes overlay videos for debugging, and finally saves compartment masks to compressed H5.

**Python dependencies**:
- `torch` + `sam2`: Core model inference and mask propagation.
- `numpy`: Mask operations and prompt-point arrays.
- `json` / `os` / `shutil` / `time` / `random`: Prompt management, file bookkeeping, and workflow control.
- `cv2`, `PIL`, `matplotlib`: Visualization and overlay rendering helpers.
- `scipy.ndimage.binary_dilation`: Mask distance and proximity checks.
- `h5py`: Final segmentation export format.
- `tqdm`: Progress tracking for prompt frame analysis.

**Expected inputs**:
- Cropped RIA frame folders (typically output from `2_crop_RIAregion.py`).
- Prompt metadata JSON containing object IDs and click coordinates.
- SAM2 model config/checkpoint paths.

**Expected outputs**:
- Per-video compartment segmentation masks saved to compressed H5.
- Optional overlay/debug videos or images showing mask quality.
- QA-derived metadata such as detected problematic frames (empty/overlap/size checks).

**External dependencies**:
- Requires Meta SAM2 repository code and compatible model weights.
- Depends on prompt JSON files being present and formatted as expected by this script.

<br>

#### <font color="#0B7285">4. Extracting brightness + orientation metadata. (`4_extract_RIAbrightness_and_orientation.py`)</font>
**General description**: This script loads segmented masks, samples background pixels, calculates per-frame brightness metrics (including background-corrected values and pixel counts), infers side/orientation from relative object centroids, and writes a wide-format CSV for downstream analysis.

**Python dependencies**:
- `h5py`: Reads segmentation masks saved in H5.
- `numpy`: Pixel-level and mask computations.
- `cv2`: Loads grayscale frame images.
- `pandas`: Builds and exports tabular brightness data.
- `scipy.ndimage.distance_transform_edt`: Finds sufficiently distant background sample locations.
- `os` / `random`: File selection and unprocessed-video selection.
- `tqdm`: Progress display for frame-by-frame extraction.

**Expected inputs**:
- Segmentation H5 files from `3_autoprompted_RIAsegmentation.py`.
- Corresponding frame images (grayscale/intensity source) for pixel sampling.

**Expected outputs**:
- Wide-format CSV containing per-frame compartment brightness values, corrected brightness, pixel counts, and orientation labels.

**External dependencies**:
- No additional third-party repo/model beyond local data products from earlier RIA steps.

<br>

#### <font color="#0B7285">5. Head segmentation for angle analysis. (`5_head_segmentation.py`)</font>
**General description**: This stage runs SAM2 on the worm head/body region using a generic point prompt, propagates masks across frames, does basic missing/empty-mask checks, and saves a dedicated head-segmentation H5 file (plus optional overlay video).

**Python dependencies**:
- `torch` + `sam2`: Video segmentation inference.
- `numpy`: Prompt arrays and mask comparisons.
- `cv2`: Frame I/O and overlay video creation.
- `h5py`: Persists masks for later head-angle extraction.
- `PIL` / `matplotlib`: Prompt and mask visualization utilities.
- `os` / `sys` / `random`: File selection and SAM2 import path wiring.

**Expected inputs**:
- Frame folders for the target video (typically the same processed/cropped stream used in prior RIA steps).
- Generic head/body point prompt coordinates.
- SAM2 model config/checkpoint paths.

**Expected outputs**:
- Head/body segmentation masks saved as an H5 file.
- Optional overlay video/images for segmentation QA.

**External dependencies**:
- Requires Meta SAM2 repository code and model checkpoints not bundled in this repo.

<br>

#### <font color="#0B7285">6. Head angle extraction + smoothing. (`6_extract_head_angle.py`)</font>
**General description**: This script converts head masks to skeletons, computes per-frame head bend angles and bend-location metrics, smooths noisy spikes, interpolates/decays through bad frames, applies side-aware sign correction, and merges the output into the final per-video CSV.

**Python dependencies**:
- `h5py`: Loads head-segmentation masks.
- `numpy` + `numpy.linalg`: Geometry, vector math, and eigendecomposition-based curvature calculations.
- `skimage.morphology`: Skeletonization.
- `scipy.ndimage`: Gaussian smoothing and curvature helpers.
- `pandas`: Stores/merges head-angle tables.
- `cv2` / `re`: Overlay video helper logic and filename frame parsing.
- `os` / `random`: File discovery and work scheduling.

**Expected inputs**:
- Head-segmentation H5 files from `5_head_segmentation.py`.
- Existing per-video CSV (typically from `4_extract_RIAbrightness_and_orientation.py`) for merge/update.

**Expected outputs**:
- Final or updated per-video CSV with head-angle and bend-location time-series columns.
- Optional diagnostic visualizations for angle/skeleton quality control.

**External dependencies**:
- No extra third-party repository/model dependency; consumes outputs from earlier RIA scripts.

### <font color="#5F3DC4">Single Worm Tracking (`/singleworm_tracking`)</font>

**General directory synopsis**: At a high level, this folder tracks one worm over time and turns movement into quantitative behavior features. The flow is: (1) convert videos into SAM-ready frames, (2) run segmentation with a full-frame pass plus a higher-definition pass, (3) extract shape/posture metrics from the cleaned masks, and then (4) analyze trajectory and movement state (forward/backward/stationary) over time. So conceptually, it goes from **raw crawling video → robust worm masks → posture + locomotion metrics**.

<br>

#### <font color="#5F3DC4">1. Video-to-frames conversion. (`1_videotoimg.py`)</font>
**General description**: Similar to the RIA converter, but optimized for crawling videos and built to optionally process frames in parallel. It checks readability, handles frame sampling/resizing, reuses existing outputs when possible, and writes SAM-ready image folders.

**Python dependencies**:
- `cv2`: Video read/write and frame extraction.
- `pathlib` / `os`: Path handling and optional CPU affinity setup.
- `multiprocessing`: Parallel frame extraction.
- `re`: Frame-number parsing from filenames.
- `tqdm`: Progress bars during multiprocessing jobs.
- `random`: Picks an unprocessed video folder.

**Expected inputs**:
- Raw single-worm crawling video files (e.g., `.avi`, `.mp4`, `.mov`, `.mkv`, `.wmv`, `.flv`).
- A source directory containing candidate videos and a target output directory for extracted frames.
- Optional processing parameters such as target FPS sampling, max-dimension resize, output image format (`jpg`/`png`), and number of worker processes.

**Expected outputs**:
- Per-video frame folders named with the source parent-folder and video stem (e.g., `parentFolder-videoName`).
- Sequentially indexed frame images (default six-digit naming like `000000.jpg`) suitable for downstream SAM2 segmentation.
- Frame-processing summary stats (new, converted, reused frames) plus any consistency warnings (missing/failed/unexpected frames).

**External dependencies**:
- No third-party repository/model dependency for this conversion step.
- Video codec support depends on your local OpenCV/FFmpeg build.
- Optional CPU affinity behavior uses OS-level support (`os.sched_setaffinity`) where available.


<br>

#### <font color="#5F3DC4">2. Two-pass autoprompted segmentation (full-frame + HD). (`2_autoprompted_segmentation.py`)</font>
**General description**: This is a large, production-style segmentation workflow. It does full-frame SAM2 segmentation with reusable prompt pools, performs extensive quality checks and mask repair/interpolation for problematic frames, then runs a second high-definition crop/segmentation pass and saves final HD mask data.

**Python dependencies**:
- `torch` + `sam2`: Core segmentation model operations.
- `numpy`: Prompt and mask arithmetic.
- `cv2`, `PIL`, `matplotlib`: Rendering prompt checks and overlay videos.
- `json` / `os` / `sys` / `shutil` / `time` / `random`: Prompt frame orchestration and file management.
- `scipy.ndimage.binary_dilation`: Distance/overlap checks.
- `multiprocessing`: Faster mask-overlay video generation.
- `pickle` / `copy`: Saving and safely mutating segmentation structures.
- `tqdm`: Progress tracking.

**Expected inputs**:
- Frame folders from `1_videotoimg.py`.
- Prompt pool/config JSON used to seed and reuse clicks across videos.
- SAM2 model config/checkpoint paths.

**Expected outputs**:
- Full-frame and HD-refined segmentation artifacts (pickle/H5 depending on stage/config).
- Repaired/interpolated masks for problematic frames.
- Optional overlay videos and QA/debug artifacts.

**External dependencies**:
- Requires Meta SAM2 repository code and downloaded checkpoints.
- Depends on local prompt metadata files expected by this workflow.

<br>

#### <font color="#5F3DC4">3. Shape analysis from HD masks. (`3_shape_analysis.py`)</font>
**General description**: This script takes HD segmentation outputs and computes a broad set of shape/bending metrics. It includes heavy skeleton cleanup logic (junction/self-touch fixes, endpoint ordering), interpolation/smoothing, spectral features, and rich visualization outputs before saving consolidated shape-analysis artifacts.

**Python dependencies**:
- `numpy`: Core numeric processing.
- `scipy` (`interpolate`, `ndimage`, `signal`, `optimize`, `spatial.distance`): Spline fitting, smoothing, FFT/welch metrics, assignment, and distance computations.
- `skimage` (`morphology`, `graph`, `measure`): Skeletonization and pathing on masks.
- `networkx`: Gap-closing shortest-path logic for broken skeleton segments.
- `matplotlib`: Diagnostic and summary plotting.
- `pickle`: Read/write intermediate analysis objects.
- `collections.defaultdict`: Graph-like neighbor bookkeeping.
- `os` / `gc` / `random` / `time`: Batch processing and memory hygiene.

**Expected inputs**:
- HD segmentation outputs from `2_autoprompted_segmentation.py`.

**Expected outputs**:
- Shape-analysis artifact files (commonly pickle-based) containing per-frame posture/curvature/shape metrics.
- Diagnostic plots/visual summaries for skeleton and metric quality.

**External dependencies**:
- No additional external repo/model requirement beyond segmentation outputs from earlier steps.

<br>

#### <font color="#5F3DC4">4. Trajectory + movement-state analysis. (`4_path_analysis.py`)</font>
**General description**: This script consumes shape-analysis outputs and computes movement trajectories, centroids, velocity-based motion states (forward/backward/stationary), head-tail correction passes, path metrics, plots, and optional labeled videos. Final per-video path-analysis files are saved for downstream stats.

**Python dependencies**:
- `numpy`: Motion math and vector calculations.
- `scipy.ndimage.center_of_mass`: Centroid extraction.
- `scipy.signal` (`savgol_filter`, `medfilt`): Path/motion smoothing.
- `scipy.spatial.distance.euclidean`: Distance-based metrics.
- `matplotlib`: Path and correction diagnostic plots.
- `cv2`: Optional movement-analysis video generation.
- `pickle`: Input/output analysis payloads.
- `os` / `random` / `time` / `traceback`: Batch orchestration and robust error reporting.

**Expected inputs**:
- Shape-analysis artifacts from `3_shape_analysis.py`.

**Expected outputs**:
- Path-analysis artifacts with trajectory, speed, and movement-state annotations.
- Optional path/motion plots and labeled movement videos.

**External dependencies**:
- No additional third-party repo/model dependency for this analysis stage.

### <font color="#1C7ED6">Droplet Swimming (`/droplet_swimming`)</font>

**General directory synopsis**: At a high level, this folder is the swim-analysis pipeline for droplet videos. The flow is: (1) convert video into frames, (2) run a full-frame segmentation pass to get initial worm locations, (3) run a high-definition crop/segmentation pass for cleaner masks, and then (4) compute swim-shape features like curvature, amplitude, wavelength, and temporal frequency trends. So conceptually, it goes from **raw swim video → refined masks → quantitative swimming kinematics**.

<br>

#### <font color="#1C7ED6">1. Video-to-frames conversion. (`1_videotoimg.py`)</font>
**General description**: This is the droplet/swimming flavor of video preprocessing. It extracts frames (with optional resize and frame-rate downsampling), handles missing/extra frame bookkeeping, and outputs clean image folders for SAM2.

**Python dependencies**:
- `cv2`: Video decoding and image export.
- `pathlib`: Path-safe output directory creation.
- `re`: Frame index parsing for consistency checks.

**Expected inputs**:
- Raw droplet/swimming video files.
- Optional resize/downsample parameters.

**Expected outputs**:
- Per-video frame folders with normalized frame names for SAM2 processing.

**External dependencies**:
- No external repo/model dependency for this conversion step.

<br>

#### <font color="#1C7ED6">2. Full-frame SAM2 segmentation. (`2_fframe_segmentation.py`)</font>
**General description**: This pass appends a generic prompt frame, runs full-frame SAM2 propagation across the video, performs basic detection-quality checks (empty/too small/too large masks), removes the helper prompt frame, and saves frame-wise segmentation to pickle.

**Python dependencies**:
- `torch` + `sam2`: Segmentation and propagation.
- `numpy`: Prompt and mask handling.
- `cv2`, `PIL`, `matplotlib`: Frame read and prompt/mask visualization.
- `pickle`: Stores per-frame segmentation dictionaries.
- `os` / `sys` / `shutil` / `pathlib`: File operations and repo-path setup.

**Expected inputs**:
- Frame folders from `1_videotoimg.py`.
- Generic prompt frame/click setup used to initialize full-frame propagation.
- SAM2 model config/checkpoint paths.

**Expected outputs**:
- Full-frame segmentation results saved as pickle dictionaries.
- Optional visualization artifacts for prompt/mask QA.

**External dependencies**:
- Requires Meta SAM2 repository code and model checkpoints not included in this repo.

<br>

#### <font color="#1C7ED6">3. High-definition swim segmentation. (`3_swim_hdsegmentation.py`)</font>
**General description**: This stage uses the full-frame results to crop around the worm and run a higher-resolution segmentation pass. It supports an intermediate large-crop fallback pass when masks are empty, then finalizes HD masks and saves either intermediate pickle or final H5.

**Python dependencies**:
- `torch` + `sam2`: HD segmentation inference.
- `numpy`: Mask center calculations and prompt arrays.
- `cv2`: Cropping/resizing and image I/O.
- `pickle` / `h5py`: Intermediate and final segmentation persistence.
- `tqdm`: Crop-loop progress bars.
- `PIL` / `matplotlib`: Prompt-frame visualization.
- `os` / `sys` / `shutil` / `pathlib`: Workflow file management.

**Expected inputs**:
- Full-frame segmentation pickle outputs from `2_fframe_segmentation.py`.
- Original frame folders for HD crop extraction.
- SAM2 model config/checkpoint paths.

**Expected outputs**:
- HD segmentation outputs (intermediate pickle and/or finalized H5, based on workflow stage).
- Optional intermediate fallback-pass artifacts when empty-mask recovery is used.

**External dependencies**:
- Requires Meta SAM2 repository and downloaded checkpoints.

<br>

#### <font color="#1C7ED6">4. Swimming shape analysis. (`4_shape_analysis.py`)</font>
**General description**: This script loads HD swim masks and computes morphology/kinematic descriptors (shape class, curvature, amplitude, wavelength, wave number, temporal frequency summaries, etc.). It includes skeleton repair logic for self-touching shapes and writes the final analysis package to H5.

**Python dependencies**:
- `numpy`: Core shape/curvature math.
- `h5py`: Reads HD segmentation and writes analysis outputs.
- `scipy` (`interpolate`, `ndimage`, `signal`, `spatial.distance`): Curve smoothing, curvature analysis, and frequency metrics.
- `skimage` (`morphology`, `graph`, `measure`): Skeleton and path extraction.
- `networkx`: Skeleton gap repair via shortest paths.
- `collections` (`Counter`, `defaultdict`): Shape-count summaries and neighbor logic.
- `pathlib` / `os`: Consistent file naming and I/O.

**Expected inputs**:
- HD swim segmentation data from `3_swim_hdsegmentation.py` (typically H5).

**Expected outputs**:
- Analysis H5 containing per-frame/per-video swimming kinematic and morphology features.

**External dependencies**:
- No additional external repo/model dependency for this post-segmentation analysis.

### <font color="#2B8A3E">Multi-Worm Feature Extraction (`/multiworm_feature_extraction`)</font>

**General directory synopsis**: At a high level, this folder handles many worms in still images (rather than one worm in one video). The flow is: (1) convert microscope TIFF images into model-friendly JPGs, (2) generate candidate masks with SAM, classify each cutout as worm/not-worm, clean/merge valid worm masks, and then extract per-worm morphology metrics for downstream analysis. So conceptually, it goes from **raw multi-worm images → filtered worm detections → per-worm feature tables**.

<br>

#### <font color="#2B8A3E">1. TIFF to JPEG conversion. (`1_convert_images.py`)</font>
**General description**: This is a straightforward converter that walks a source directory tree, rescales 16-bit TIFF images to 8-bit, and mirrors the folder structure while saving JPEG outputs for downstream model processing.

**Python dependencies**:
- `tifffile`: Reads high bit-depth TIFF images.
- `numpy`: Pixel rescaling math from 16-bit to 8-bit.
- `PIL`: Writes converted JPEG images.
- `os`: Recursive directory traversal and output structure creation.

**Expected inputs**:
- Source directory tree of microscopy TIFF images.

**Expected outputs**:
- Mirrored output directory tree containing converted JPEG images.

**External dependencies**:
- No third-party repository/model dependency for this conversion utility.

<br>

#### <font color="#2B8A3E">2. SAM cutouts + worm classifier + metrics extraction. (`2_extract_wormcutouts.py`)</font>
**General description**: This is the large multi-worm pipeline. It uses SAM2 automatic mask generation, filters edge artifacts, creates mask cutouts, classifies each cutout with the fine-tuned ViT worm/not-worm model, merges/cleans accepted worm masks, extracts morphology metrics per worm, and saves cutouts + metrics outputs for each image.

**Python dependencies**:
- `torch` / `torchvision` / `torch.nn`: Loads and runs the ViT-based worm classifier.
- `sam2` (`build_sam2`, `SAM2AutomaticMaskGenerator`): Automatic segmentation candidate generation.
- `numpy`: Mask and metric math.
- `cv2`: Contours, connected components, masking, and image I/O.
- `PIL`: Classifier input loading.
- `skimage.measure` / `skimage.morphology` / `skimage.graph`: Region labeling and medial-axis/route computations.
- `scipy.ndimage.convolve`: Branch/end point detection on skeletonized structures.
- `matplotlib`: Optional segmentation visual checks.
- `glob` / `os` / `sys` / `pickle` / `shutil`: Batch file orchestration and persistence.

**Expected inputs**:
- Multi-worm images (commonly JPEG outputs from `1_convert_images.py`, though other supported image sources may work).
- SAM2 configuration/checkpoint assets for automatic mask generation.
- Fine-tuned worm classifier weights/checkpoint (ViT model).

**Expected outputs**:
- Per-image worm cutout files and filtered/merged worm masks.
- Per-image or aggregated morphology metrics tables/artifacts.
- Optional QA visualizations for accepted/rejected candidates.

**External dependencies**:
- Requires Meta SAM2 repository code and compatible SAM2 checkpoints.
- Requires the external worm classifier model weights from Hugging Face: https://huggingface.co/lillyguisnet/celegans-classifier-vit-h-14-finetuned.
- Initial model download may require internet access unless weights are already cached locally.