# Assignment 3 – AI Imaging Coding Case Study

## Overview

This project implements a complete biomedical image-analysis pipeline for
fluorescence microscopy images of cell nuclei.

The pipeline combines data preprocessing, exploratory data analysis (EDA),
a Vision Language Model (VLM), classical image segmentation, U-Net deep
learning segmentation, and a hybrid structured reporting approach.

The main purpose of the project is to compare different image-analysis
methods and investigate their reliability for biomedical imaging.

This project is for educational purposes only and does not provide
clinical diagnoses.


## Dataset

The dataset contains 116 paired microscopy images and masks.

Dataset split:

- Training images: 80
- Validation images: 20
- Test images: 16
- Total images: 116

All images are converted to grayscale, normalised to the range 0–1 and
resized to 256 × 256 pixels before analysis.


## Requirements

Python 3 is required.

Install the required Python packages using:

pip install numpy pandas matplotlib pillow requests scikit-image torch

Ollama is also required for the language-model components.

Required Ollama models:

ollama pull llama3.2-vision
ollama pull llama3.2


## Starting Ollama

Before running the pipeline, start the Ollama server in a Terminal:

ollama serve

Keep this Terminal window open.

Open another Terminal window and confirm that the models are installed:

ollama list

The following models should appear:

llama3.2-vision
llama3.2


## Running the Project

Navigate to the folder containing the Python file.

Example:

cd "/path/to/Assignment 3"

Run the pipeline using:

python assignment3_complete_pipeline.py

The program automatically downloads and extracts the dataset if it is
not already available.


## Task 1 – Data Pre-processing and Vision Language Model

The first task prepares the microscopy images for analysis.

The preprocessing stage:

1. Reads the microscopy images.
2. Converts colour images to grayscale.
3. Normalises pixel intensities.
4. Resizes images to 256 × 256 pixels.
5. Generates representative sample images.
6. Generates an intensity histogram.

The Vision Language Model stage uses Llama 3.2 Vision through Ollama.

Two prompts are included.

Simple prompt:

"Describe this biomedical image."

The improved prompt restricts the model to visible image characteristics,
prevents diagnostic claims and requests structured JSON output.

The JSON contains:

- modality
- tissue_type
- notable_features
- image_quality

The prompt also instructs the model to use "uncertain" when there is not
enough visual evidence.

In the recorded experiment, the VLM returned an HTTP 500 internal server
error. Therefore, no valid VLM image description was used as an
experimental result. The error was recorded rather than replacing it
with fabricated output.


## Task 2 – Classical Image Processing

Classical segmentation is performed using Otsu thresholding.

After thresholding, morphological operations are used to:

- remove small objects
- remove small holes
- perform binary opening
- perform binary closing

Connected regions are then identified and quantitative features are
extracted.

The extracted features include:

- area
- eccentricity
- solidity
- mean intensity
- perimeter
- major axis length
- minor axis length

The measured results for the representative image were:

n_objects = 9
foreground_fraction = 0.0196
area_mean = 142.44
area_median = 115.00
area_std = 57.14
eccentricity_mean = 0.504
solidity_mean = 0.956
mean_intensity_mean = 0.744
perimeter_mean = 42.47
major_axis_mean = 14.97
minor_axis_mean = 11.92

These numerical measurements are passed to the text model rather than
giving the model direct access to the image.

The model is instructed not to invent visual information and returns
structured JSON containing:

- n_objects
- density_class
- shape_regularity
- quality_flag
- description

This numbers-first approach makes the generated description easier to
audit because it is based on measured image features.


## Task 3 – U-Net Segmentation

A compact U-Net was implemented using PyTorch for nuclei segmentation.

Three loss functions are compared:

1. Binary Cross Entropy (BCE)
2. Dice loss
3. BCE + Dice loss

The default training configuration is:

Epochs: 15
Batch size: 4
Learning rate: 0.001

The model is evaluated using:

- Dice coefficient
- Intersection over Union (IoU)

The best-performing U-Net used Dice loss.

Final validation results:

U-Net Dice = 0.7146
U-Net IoU = 0.5581

For comparison, classical Otsu segmentation achieved:

Otsu Dice = 0.6999
Otsu IoU = 0.5405

Therefore, the U-Net achieved slightly better overall validation
performance than Otsu thresholding.


## U-Net vs Otsu Examples

An example where U-Net performed better:

Image: val_005.png

U-Net Dice = 0.7359
Otsu Dice = 0.7113

An example where the U-Net advantage was very small:

Image: val_018.png

U-Net Dice = 0.7637
Otsu Dice = 0.7603

These results show that deep learning improved the overall segmentation
performance, although classical thresholding remained competitive on
some images.


## Task 4 – Hybrid Pipeline

The final stage combines U-Net segmentation, quantitative feature
extraction and structured language-model reporting.

The best U-Net model is applied to the test images.

For each image, the pipeline:

1. Loads and preprocesses the image.
2. Generates a U-Net segmentation mask.
3. Extracts region measurements.
4. Calculates quantitative image statistics.
5. Creates a structured record.
6. Generates a short narrative description.
7. Saves the results to JSON and CSV files.

Important quantitative fields are preserved directly from the measured
data rather than allowing the language model to modify them.

The structured output includes:

- image_id
- n_objects
- mean_area
- density_class
- quality_flag

Additional measurements include:

- foreground_fraction
- mean_eccentricity
- mean_solidity

All 16 test images were processed by the hybrid pipeline.

The combined results are saved as:

task4/test_hybrid_records.csv


## Output Structure

After the pipeline finishes, the main results are stored inside:

assignment3_run/outputs/

The output structure is approximately:

assignment3_run/
│
├── dataset/
│
└── outputs/
    │
    ├── dataset_counts.json
    │
    ├── REPORT_RESULTS_SUMMARY.md
    │
    ├── task1/
    │   ├── eda_sample_images.png
    │   ├── eda_intensity_histogram.png
    │   └── task1_vlm_results.json
    │
    ├── task2/
    │   ├── classical_segmentation.png
    │   ├── regionprops.csv
    │   └── task2_numbers_first.json
    │
    ├── task3/
    │   ├── evaluation_metrics.csv
    │   ├── unet_loss_curves.png
    │   ├── unet_dice_curves.png
    │   ├── best_unet_model.pt
    │   ├── unet_vs_otsu_per_image.csv
    │   ├── unet_vs_otsu_examples.json
    │   └── validation_panels/
    │
    └── task4/
        ├── test_hybrid_records.csv
        ├── *_hybrid_mask.png
        ├── *_hybrid_record.json
        └── *_unet_regionprops.csv


## Main Results

Dataset:

Total images = 116
Training = 80
Validation = 20
Test = 16

Classical segmentation representative result:

Detected objects = 9
Foreground fraction = 0.0196
Mean object area = 142.44
Mean eccentricity = 0.504
Mean solidity = 0.956

Segmentation comparison:

Best U-Net loss = Dice
U-Net Dice = 0.7146
U-Net IoU = 0.5581
Otsu Dice = 0.6999
Otsu IoU = 0.5405

The U-Net therefore produced the best overall validation performance,
although the improvement over Otsu was relatively small.


## Reliability and Limitations

The pipeline was designed to keep quantitative measurements as the main
source of information.

The language models are prevented from making clinical diagnoses and are
instructed to use uncertainty when there is insufficient evidence.

However, several limitations remain:

- The dataset is relatively small.
- Segmentation performance is not perfect.
- Blur and low contrast can reduce segmentation quality.
- Touching nuclei can be difficult to separate.
- Language models can produce unsupported interpretations.
- The VLM experienced an HTTP 500 error during the recorded run.
- Results have not been externally validated on an independent clinical
  dataset.

For these reasons, this system should not be used for clinical
decision-making.


## Reproducibility

A fixed random seed of 42 is used to improve reproducibility.

The default configuration is:

IMAGE_SIZE = 256
SEED = 42
EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 0.001

The pipeline also saves intermediate measurements, model results, JSON
records, CSV files and plots so that the analysis can be inspected.


## References

Ronneberger, O., Fischer, P. and Brox, T. (2015).
U-Net: Convolutional Networks for Biomedical Image Segmentation.
MICCAI.

Otsu, N. (1979).
A Threshold Selection Method from Gray-Level Histograms.
IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.

Van der Walt, S. et al. (2014).
scikit-image: Image Processing in Python.
PeerJ, 2, e453.

Paszke, A. et al. (2019).
PyTorch: An Imperative Style, High-Performance Deep Learning Library.
NeurIPS.


## Final Note

This project is an educational biomedical image-analysis case study.
The generated results and language-model outputs are intended for
technical evaluation only and should not be interpreted as medical or
clinical advice.
