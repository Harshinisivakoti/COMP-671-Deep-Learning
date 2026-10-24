# Plant Disease Recognition Using Vision Transformer with Attention Analysis

## Project overview

This project studies plant disease recognition from leaf images. It
compares three experiments:

1.  A basic Convolutional Neural Network (CNN) used as the baseline.
2.  A pretrained Vision Transformer (ViT-B/16) trained with image
    augmentation.
3.  A separately trained pretrained ViT-B/16 without random image
    augmentation.

The project also uses attention rollout to create heatmaps over example
images and examines incorrect predictions to better understand the
model's limits.

The notebook was developed in Google Colab with Python, PyTorch,
Torchvision, and a GPU runtime. The notebook is the main source file for
the complete workflow.

## Project goals

-   Prepare plant image crops from the dataset's YOLO labels.
-   Select the classes available in all three data splits.
-   Train and evaluate a CNN baseline.
-   Train and evaluate a pretrained Vision Transformer.
-   Compare the ViT results with and without random augmentation.
-   Review attention maps and examples of incorrect predictions.
-   Record the steps and settings so the work can be checked and
    repeated.

## Dataset

**Dataset name:** PlantDoc\
**Kaggle page:** https://www.kaggle.com/datasets/andresmgs/plantdec

Download the dataset from Kaggle. The notebook expects a ZIP archive
containing the `train`, `valid`, and `test` folders, their `images` and
`labels` subfolders, and the `data.yaml` file. The notebook's upload
cell asks you to select the ZIP archive from your computer.

The dataset archive used in this project includes YOLO-format labels.
These labels are used to crop plant regions before image classification.

### Dataset preparation in this notebook

The notebook performs the following steps:

1.  Uploads the dataset ZIP file to the Colab session.
2.  Extracts it to `/content/plantdoc_raw`.
3.  Reads the class names from `data.yaml`.
4.  Uses the YOLO labels to crop plant regions into class folders.
5.  Skips missing or empty labels, invalid boxes, very small crops, and
    images with more than one class.
6.  Keeps only classes with images in the training, validation, and test
    splits.
7.  Resizes images to `224 × 224` pixels and normalizes them using
    ImageNet mean and standard deviation.
8.  Creates PyTorch data loaders with a batch size of 32.

The final filtered dataset contains **27 classes** and these numbers of
images:

  Split             Images
  ------------ -----------
  Training           6,898
  Validation         1,245
  Test                 454
  **Total**      **8,597**

The exact counts are produced by the notebook. If your downloaded
dataset version differs, your counts may also differ. Use the outputs
from your run when reporting results.

## Repository contents

A simple repository structure is:

``` text
plant-disease-recognition/
├── README.md
└── plant_disease_vit_attention Final.ipynb
```

Keep the notebook filename consistent with the file you upload to
GitHub, or update the name in this README. If you store the notebook
under a folder such as `notebooks/`, update the path above to match your
repository.

The notebook contains data preparation, model definitions, training,
testing, comparison, attention rollout, and failure analysis. Dataset
images and trained model files are created in the temporary Colab
runtime; they are not automatically saved to GitHub.

## Requirements and environment

The recorded environment for the completed run was:

-   Google Colab
-   Python 3.13.15
-   PyTorch 2.11.0+cu130
-   Torchvision 0.26.0+cu130
-   CUDA available
-   NVIDIA Tesla T4 GPU

The notebook also uses these Python packages:

-   `PyYAML` (`yaml`) to read `data.yaml`
-   `Pillow` (`PIL`) to read and crop images
-   `matplotlib` to display images and attention maps
-   `scikit-learn` to calculate classification metrics
-   Python standard libraries such as `os`, `zipfile`, `pathlib`,
    `shutil`, and `collections`

Google Colab normally includes PyTorch, Torchvision, Matplotlib, Pillow,
and scikit-learn. First run the notebook's environment-check cell.
Install a missing package only if Colab reports that it is unavailable:

``` python
%pip install pyyaml pillow matplotlib scikit-learn
```

Do not reinstall PyTorch or Torchvision without a reason. Their versions
must be compatible with the Colab runtime and its CUDA setup.

## How to run the project in Google Colab

### Step 1: Open the notebook

1.  Download or clone this repository, or open the notebook from the
    GitHub repository.
2.  Go to https://colab.research.google.com/.
3.  Select **File → Upload notebook** to upload the `.ipynb` file, or
    open the notebook from GitHub.
4.  In Colab, select **Runtime → Change runtime type**.
5.  Select **T4 GPU** if it is available, then save the setting.

### Step 2: Check the environment

Run the first code cell. It prints the Python, PyTorch, and Torchvision
versions and checks whether CUDA is available.

The completed run used an NVIDIA Tesla T4 GPU. If a GPU is unavailable,
training may be much slower on a CPU.

### Step 3: Download and upload the dataset ZIP

1.  Open the Kaggle dataset page:
    https://www.kaggle.com/datasets/andresmgs/plantdec.
2.  Download the dataset as a ZIP file.
3.  Return to the notebook and run the cell that uses `files.upload()`.
4.  Select the downloaded ZIP archive from your computer.

The notebook's upload cell saves the uploaded archive in the current
Colab session. The next cell finds the ZIP file and extracts it to:

``` text
/content/plantdoc_raw
```

The extracted archive should contain `train`, `valid`, and `test`
directories, each with `images` and `labels` folders, along with
`data.yaml`. If your ZIP has an extra top-level folder or a different
layout, check the extracted folder listing and adjust the paths before
continuing.

### Step 4: Prepare classification crops

Run the cells under the notebook heading
**`#Prepare classification crops`** in order.

These cells read the YOLO labels, crop plant regions, organize crops by
class, and remove unusable samples. The crop output is saved to:

``` text
/content/plantdoc_classification
```

The next cell finds classes that have images in every split and creates
the filtered dataset at:

``` text
/content/plantdoc_classification_filtered
```

Check the printed class count and image counts. The completed run used
27 classes, with 6,898 training images, 1,245 validation images, and 454
test images.

### Step 5: Load and check the images

Run the preprocessing and DataLoader cells. The notebook resizes images
to `224 × 224`, converts them to tensors, and normalizes them with
ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`.

Training images use random horizontal flips and random rotations of up
to 10 degrees. Validation and test images use resizing and normalization
without these random changes. The batch size is 32.

The notebook checks that the class mapping is the same across the
training, validation, and test datasets. It also displays a batch of
example images.

### Step 6: Train and evaluate the CNN baseline

Run the notebook sections in order:

-   **`# Build the CNN baseline`**
-   **`##Train the CNN`**
-   **`#Evaluate the best CNN on the test set  ---`**

The CNN uses four convolutional layers, ReLU activation, max pooling,
dropout, and a final classification layer for 27 classes. It is trained
for 10 epochs using Adam with a learning rate of `0.001` and
cross-entropy loss.

The best checkpoint is saved in the active Colab session as:

``` text
/content/best_cnn_model.pth
```

The test section reports accuracy and a classification report.

### Step 7: Train and evaluate the ViT with augmentation

Run these notebook sections in order:

-   **`## Build the Vision Transformer (ViT)`**
-   **`#Train the Vision Transformer`**
-   **`## Evaluate the best ViT on the test set`**

The model starts from the pretrained Torchvision `vit_b_16` weights. Its
final classification head is replaced for the 27 project classes. The
model is fine-tuned for five epochs with AdamW, learning rate `0.0001`,
weight decay `0.01`, and cross-entropy loss.

The best checkpoint is saved as:

``` text
/content/best_vit_model.pth
```

The test section reports accuracy, precision, recall, F1-scores, and the
class-level classification report.

### Step 8: Compare the CNN and ViT

Run the notebook section:

-   **`## comparison between the CNN and ViT`**

This section compares the model results. In the recorded run, the CNN
test accuracy was 17.84%, while the augmented ViT test accuracy was
61.45%. Use the metrics printed by your own run if you retrain the
models.

### Step 9: Run the no-augmentation experiment

Run the sections in order:

-   **`# data-augmentation experiment ## ViT trained without random augmentation.`**
-   **`## Train ViT without augmentation`**
-   **`## Evaluate the no-augmentation ViT on the test set`**

This experiment removes random horizontal flips and rotations from
training. Resizing and normalization remain in place. It trains a
separate pretrained ViT for five epochs with AdamW, learning rate
`0.0001`, and weight decay `0.01`.

The checkpoint is saved as:

``` text
/content/best_vit_noaug_model.pth
```

The recorded test accuracy was 53.96%, compared with 61.45% for the
augmented ViT. Since the two ViT runs were initialized separately, this
difference should not be treated as proof that augmentation alone caused
the entire improvement.

### Step 10: Review augmented ViT accuracy

Run the section:

-   **`##augmented ViT accuracy`**

This section loads the saved augmented ViT and reports its test metrics
and class-level performance.

### Step 11: Create attention rollout visualizations

Run the section:

-   **`## Implement attention rollout`**

The attention rollout code collects attention values from the ViT
Transformer blocks, combines them, and creates a heatmap over an image.
These maps help inspect image regions highlighted by the model. They do
not prove why a prediction was made.

If the notebook contains a separate cell to recreate and save the
visualization, run that cell after the attention-analysis cell, while
the required variables and model are available.

### Step 12: Run failure analysis

Run the section:

-   **`## failure-analysis`**

This section reviews incorrect predictions from the augmented ViT,
records true and predicted class IDs and confidence scores, and creates
a figure and CSV file.

The failure-analysis outputs are saved to:

``` text
/content/failure_analysis/misclassified_predictions.csv
/content/failure_analysis/misclassified_attention_analysis.png
```

In the recorded run, the model correctly classified 279 of 454 test
images and incorrectly classified 175. Some incorrect predictions had
confidence above 99%. Review these examples rather than assuming that
high confidence means the prediction is correct.

## Results from the recorded run

  Experiment                   Test accuracy   Weighted F1-score
  -------------------------- --------------- -------------------
  CNN baseline                        17.84%               13.7%
  ViT with augmentation               61.45%               60.7%
  ViT without augmentation            53.96%               52.8%

The augmented ViT had the highest test accuracy in this run. The CNN
result was much lower. These results apply to this dataset split and
training setup; they do not establish how the model will perform on
every crop, farm, or growing condition.
## Model Comparison and Evaluation Results

This project compares three deep learning models for plant disease recognition using leaf images:

1. A Convolutional Neural Network (CNN) used as the baseline model.
2. A pretrained Vision Transformer (ViT-B/16) trained with random image augmentation.
3. A separately trained pretrained Vision Transformer (ViT-B/16) without random image augmentation.

### Test Results

| Model | Test Accuracy |
|---|---:|
| CNN baseline | 17.84% |
| ViT with random augmentation | 61.45% |
| ViT without random augmentation | 53.96% |

The ViT trained with random image augmentation achieved the highest test accuracy at 61.45%. This was 43.61 percentage points higher than the CNN baseline and 7.49 percentage points higher than the ViT trained without random augmentation.

The notebook also evaluates precision, recall, and F1-score to provide a broader view of classification performance.

| Evaluation Metric | CNN Baseline | ViT with Augmentation | ViT without Augmentation |
|---|---:|---:|---:|
| Macro Precision | 0.111 | 0.615 | 0.550 |
| Macro Recall | 0.148 | 0.620 | 0.548 |
| Macro F1-score | 0.103 | 0.595 | 0.520 |
| Weighted Precision | 0.146 | 0.652 | 0.574 |
| Weighted Recall | 0.178 | 0.615 | 0.540 |
| Weighted F1-score | 0.137 | 0.607 | 0.528 |

These results show that the ViT models performed better than the CNN baseline in this experiment. However, performance varied across plant disease classes, and some classes remained difficult to identify correctly.

The augmentation experiment should be interpreted carefully. The two ViT models were trained separately, so differences in their results may reflect random initialization and training variation as well as the use of augmentation.

## Attention Heatmaps and Failure Analysis

### Attention Heatmaps

The notebook implements attention rollout to examine where the Vision Transformer directs attention in an image. It combines attention information from transformer layers and maps the resulting values onto the input image to create a heatmap.

To generate attention heatmaps:

1. Open the project notebook in Google Colab.
2. Upload the PlantDoc dataset and run the notebook cells in order.
3. Complete the model training or load the saved ViT checkpoint, if available.
4. Run the **Implement attention rollout** section.
5. Inspect the generated heatmaps and image overlays.

The heatmaps provide a visual aid for examining model predictions. They should not be treated as proof that the highlighted regions fully explain the model's decision.

### Failure Analysis

The notebook examines incorrect predictions made by the ViT model on the test set. This analysis helps identify classes that may be visually similar or difficult for the model to distinguish.

The test set contains 454 images. The model classified 279 images correctly and 175 incorrectly, resulting in a test accuracy of 61.45%.

The failure-analysis table records the test image index, true class ID, predicted class ID, prediction confidence, and whether the prediction was correct.

#### Examples of Misclassified Predictions

| Test Index | True Class ID | Predicted Class ID | Prediction Confidence |
|---:|---:|---:|---:|
| 245 | 24 | 25 | 0.9978 |
| 33 | 12 | 13 | 0.9978 |
| 82 | 15 | 14 | 0.9974 |
| 83 | 15 | 14 | 0.9971 |
| 84 | 15 | 14 | 0.9959 |
| 81 | 15 | 10 | 0.9878 |

All six predictions shown above were incorrect. In particular, class 15 was repeatedly predicted as class 14, suggesting that these classes may be difficult for the model to distinguish.

The examples also show that high prediction confidence does not guarantee a correct classification. Therefore, confidence values should be considered alongside actual predictions and evaluation metrics.

#### How to Generate Failure-Analysis Outputs

1. Run the data preparation and model evaluation cells in the notebook.
2. Execute the **failure-analysis** section.
3. Review the generated misclassification table and selected image examples.
4. Examine the CSV file and attention-analysis visualization.

The notebook saves the failure-analysis outputs under `/content/failure_analysis/` in Google Colab:

- `misclassified_predictions.csv` — contains the recorded misclassified predictions.
- `misclassified_attention_analysis.png` — visualizes selected misclassified examples for further inspection.

Files saved under `/content/` are temporary and may be lost when the Colab runtime is reset. Download these files or copy them to Google Drive if you want to preserve them.

### Limitations

This project has several limitations. The test accuracy indicates that the model still makes a substantial number of incorrect predictions. Visually similar disease classes may be difficult to distinguish, and high-confidence errors can occur. Attention heatmaps support visual inspection but do not establish the exact reasoning behind a prediction. The model should therefore be treated as an experimental classification tool rather than a replacement for expert plant disease diagnosis.

## Important notes about rerunning

-   Run cells in notebook order. Later cells depend on variables,
    datasets, and checkpoints created earlier.
-   Google Colab storage under `/content` is temporary. Files can
    disappear when the runtime disconnects or resets.
-   If the runtime resets, upload the dataset ZIP again and rerun the
    required cells. If you need the final analysis, rerun the relevant
    training and evaluation cells to recreate the checkpoints.
-   The pretrained ViT may need internet access to download its weights
    the first time it is initialized.
-   Training may produce slightly different results because of random
    initialization, data order, augmentation, and runtime differences.
-   The notebook uses the same held-out test split for model comparison.
    Do not tune model settings repeatedly based on test results; use
    validation results during model development.
-   The code saves model weights in the Colab session. Download any
    `.pth`, `.csv`, or `.png` output that you want to preserve, or
    upload it to your repository when appropriate.
-   The dataset itself is not included in this repository. Download it
    from Kaggle and follow the dataset's license and attribution terms.

## Reproducibility and GitHub commit history

Keep the notebook and this README in the repository. Make meaningful
commits as you develop the project instead of uploading everything in
one commit. For example, commits can cover the initial notebook, data
preparation, CNN baseline, ViT training, augmentation comparison,
attention rollout, failure analysis, and documentation. Use commit
messages that describe the changes you actually made.

Git command examples, if you use Git locally:

``` bash
git add README.md
git commit -m "Add project README and run instructions"
git push
```

If you edit directly on GitHub, use the commit message box each time you
make a meaningful change. Confirm that the repository shows the full
commit history.

## Limitations and responsible use

The model still makes incorrect predictions, and some classes are harder
to identify than others. Attention maps are useful for inspection but do
not provide a complete explanation of the model's decisions. The dataset
may not represent all crops, regions, disease stages, lighting
conditions, or farm settings. This project did not test the model in
real farming conditions or measure its real-world safety, privacy risk,
or energy use.

Do not use the model as the only basis for plant treatment decisions.
Important predictions should be checked against reliable agricultural
guidance or by an agricultural expert.

## Credits

-   **Dataset:** PlantDoc dataset on Kaggle:
    https://www.kaggle.com/datasets/andresmgs/plantdec
-   **Tools:** Google Colab, Python, PyTorch, and Torchvision.

Check the dataset page and its included license file for the required
dataset attribution and use conditions.

## License

The source code in this repository does not automatically change the
license of the PlantDoc dataset or its images. Review the dataset's
current license and attribution requirements before redistributing data
or derived files. Add a separate license for your own code only if you
intend to license it for reuse.
