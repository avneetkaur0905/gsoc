# Assignment Plan

This document explains:

- what this assignment is asking
- what we are doing right now
- what is already done
- what still needs to be done

## 1. Big Picture

You have 2 tasks.

### Task 1: Classification

Goal:
Build a model that looks at a painting image and predicts things like:

- style
- artist
- genre

Simple meaning:
"Given one painting, tell me what kind of painting it is."

### Task 2: Similarity

Goal:
Build a model that finds paintings that are similar to each other.

Simple meaning:
"Given one painting, show me other paintings that look similar."

## 2. Our Agenda

We are doing the work in this order:

1. Set up the workspace and software
2. Inspect the datasets
3. Check what data is actually available
4. Start with simple baselines
5. Build stronger models
6. Evaluate results
7. Prepare notebooks and submission material

## 3. What We Have Already Done

### Workspace setup

Created:

- `src/task1/`
- `src/task2/`
- `notebooks/task1_classification.ipynb`
- `notebooks/task2_similarity.ipynb`
- `reports/`
- `artifacts/`
- `README.md`
- `requirements.txt`

### Software setup

Installed a local Python environment in:

- `.venv/`

Installed the main tools:

- Jupyter
- pandas
- scikit-learn
- torch
- torchvision
- timm
- faiss
- open_clip

### Dataset inspection

#### Task 1

We checked the Style split from ArtGAN and confirmed:

- `style_train.csv` loads
- `style_val.csv` loads
- there are many rows

But we also found an important problem:

- the actual WikiArt image files are not present in the expected local path yet

That means:

- Task 1 metadata is ready
- Task 1 images are missing
- Task 1 model training cannot start yet

#### Task 2

We checked the National Gallery open data and confirmed:

- `objects.csv` loads
- `published_images.csv` loads
- the merge between artwork metadata and image metadata works

We found:

- merged image records: `127109`
- painting records: `4053`

That means Task 2 is in a usable state and can move forward.

## 4. Where We Are Right Now

Current status:

- Task 1: blocked by missing WikiArt image files
- Task 2: ready for the next filtering and exploration step

So the practical strategy is:

1. Continue Task 2 now
2. Solve the missing WikiArt image issue for Task 1 in parallel

## 5. What Still Needs To Be Done

### Task 1 remaining work

1. Get the actual WikiArt image dataset
2. Place the images in the correct local structure
3. Verify image paths from the CSV files
4. Build a simple baseline classifier
5. Build the required CNN + RNN classifier
6. Evaluate style / artist / genre performance
7. Find outlier examples

### Task 2 remaining work

1. Filter only open-access paintings
2. Inspect image URLs and missing values
3. Build a simple similarity baseline
4. Create image embeddings
5. Retrieve top-k similar paintings
6. Evaluate retrieval quality
7. Save examples and plots

## 6. Immediate Next Steps

### Immediate next step for Task 2

In `task2_similarity.ipynb`, do this:

1. keep only open-access paintings
2. inspect sample titles and image URLs
3. check missing values

This helps us prepare the final image dataset for similarity search.

### Immediate next step for Task 1

In `task1_classification.ipynb`, do this:

1. inspect label counts
2. inspect class names
3. do not train yet

Reason:
the real image files are still missing.

## 7. Final Goal

At the end, your assignment submission should contain:

- working notebooks for both tasks
- clear explanations
- model results
- plots and examples
- exported PDFs of the notebooks
- code in your repo

## 8. Simple Summary

If you forget everything else, remember this:

- Task 1 = classify paintings
- Task 2 = find similar paintings
- Task 1 is waiting for real WikiArt images
- Task 2 is ready to continue now
