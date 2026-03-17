# Task 1 Roadmap Execution Diagram (ASCII)

## Full Task 1 Flow

```text
+-----------------------------------------------------------+
| Task 1 Goal                                               |
| Classify paintings by Style, Genre, Artist                |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Step 1                                                    |
| Load ArtGAN split files                                   |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Step 2                                                    |
| Connect split rows to real WikiArt images                 |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Step 3                                                    |
| Create small working subset                               |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Step 4                                                    |
| Build baseline model                                      |
| ResNet18                                                  |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Step 5                                                    |
| Train and evaluate baseline                               |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Step 6                                                    |
| Build CNN + RNN model                                     |
| ResNet18 + GRU                                            |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Step 7                                                    |
| Train and evaluate CNN + RNN                              |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Step 8                                                    |
| Compare both models                                       |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Step 9                                                    |
| Error analysis / outliers                                 |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Step 10                                                   |
| Write interpretation                                      |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Repeat same pipeline for Genre                            |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Repeat same pipeline for Artist                           |
+-----------------------------------------------------------+
```

## Current Progress View

```text
[DONE]    Load Style Data
   |
   v
[DONE]    Check Images
   |
   v
[DONE]    Build Baseline
   |
   v
[DONE]    Train Baseline
   |
   v
[DONE]    Build CNN + RNN
   |
   v
[DONE]    Train CNN + RNN
   |
   v
[CURRENT] Compare Models
   |
   v
[NEXT]    Confusion Matrix + Outliers
   |
   v
[NEXT]    Write Summary
```

## Status Meaning

- `DONE` = already completed for `Style`
- `CURRENT` = this is the main step to finish now
- `NEXT` = immediate follow-up steps

## What To Do Next

1. Compare baseline vs CNN + RNN cleanly
2. Run confusion matrix
3. Inspect a few wrong predictions / outliers
4. Write a short interpretation
5. Repeat the same notebook flow for `Genre`
6. Repeat the same notebook flow for `Artist`
