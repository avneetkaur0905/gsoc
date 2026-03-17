# ArtExtract GSoC Evaluation Submission

This repository contains my completed submission for the ArtExtract prospective GSoC evaluation tasks described in [assignment.txt](/Users/avneet/project/assignment/assignment.txt).

## Submission Contents

### Task 1: Convolutional-Recurrent Architectures
- Final notebook: [notebooks/task1_final.ipynb](/Users/avneet/project/assignment/notebooks/task1_final.ipynb)
- PDF export: [reports/task1_final.pdf](/Users/avneet/project/assignment/reports/task1_final.pdf)
- Summary metrics: [reports/task1_all_results.csv](/Users/avneet/project/assignment/reports/task1_all_results.csv)

Task 1 evaluates two architectures on the WikiArt dataset across three classification attributes:
- Style
- Artist
- Genre

The two compared models are:
- ResNet-18 baseline CNN
- ResNet-18 + GRU CNN+RNN

The notebook includes:
- architecture and training strategy discussion
- metric selection and justification
- comparison across all three classification tasks
- confusion analysis and outlier analysis
- discussion of limitations and next steps

### Task 2: Similarity Retrieval
- Final notebook: [notebooks/task2_final.ipynb](/Users/avneet/project/assignment/notebooks/task2_final.ipynb)
- PDF export: [reports/task2_final.pdf](/Users/avneet/project/assignment/reports/task2_final.pdf)

Task 2 builds a similarity retrieval system for National Gallery of Art open-access paintings using:
- ResNet-18 feature embeddings
- FAISS cosine-similarity search
- qualitative retrieval analysis
- portrait-subset retrieval examples
- proxy evaluation metrics for retrieval quality
- failure-case analysis

## Repository Layout

- [assignment.txt](/Users/avneet/project/assignment/assignment.txt): original task statement
- [notebooks](/Users/avneet/project/assignment/notebooks): final submission notebooks
- [reports](/Users/avneet/project/assignment/reports): exported PDFs, figures, and result tables
- [artifacts](/Users/avneet/project/assignment/artifacts): trained checkpoints, embeddings, and intermediate outputs used by the notebooks
- [eval_style.py](/Users/avneet/project/assignment/eval_style.py) and [retrain_style.py](/Users/avneet/project/assignment/retrain_style.py): supporting experiment scripts
- [start_jupyter.command](/Users/avneet/project/assignment/start_jupyter.command): helper launcher for the project environment

## Key Results

### Task 1
From [reports/task1_all_results.csv](/Users/avneet/project/assignment/reports/task1_all_results.csv):

| Task | Model | Top-1 | Top-3 | Macro F1 |
|------|-------|-------|-------|----------|
| Style | ResNet18 (CNN) | 0.4733 | 0.7613 | 0.3585 |
| Style | ResNet18+GRU (CNN+RNN) | 0.3960 | 0.6700 | 0.2410 |
| Artist | ResNet18 (CNN) | 0.8140 | 0.9313 | 0.7909 |
| Artist | ResNet18+GRU (CNN+RNN) | 0.6387 | 0.8367 | 0.5953 |
| Genre | ResNet18 (CNN) | 0.7287 | 0.9287 | 0.6786 |
| Genre | ResNet18+GRU (CNN+RNN) | 0.6927 | 0.9060 | 0.6195 |

Overall, the baseline CNN outperformed the CNN+RNN on all three classification tasks.

### Task 2
The final similarity notebook reports:
- 2,838 embedded NGA paintings indexed for retrieval
- 512-dimensional ResNet embeddings
- FAISS exact cosine-similarity search
- medium consistency@5 of 0.524
- mean temporal gap@5 of 76.6 years

## Reproducibility Note

This repository intentionally excludes the raw `datasets/` directory because the local datasets are too large for GitHub.

What evaluators can review directly in this repo:
- the full final notebooks with outputs
- exported PDF versions of both notebooks
- generated figures and result tables in [reports](/Users/avneet/project/assignment/reports)
- trained checkpoints and derived artifacts in [artifacts](/Users/avneet/project/assignment/artifacts)

What is needed to rerun end-to-end:
- the WikiArt / ArtGAN data referenced by Task 1
- the National Gallery of Art open data referenced by Task 2
- the Python dependencies in [requirements.txt](/Users/avneet/project/assignment/requirements.txt)

In other words, the submission is fully reviewable from the committed outputs, while full reruns require the external datasets described in the original assignment.
