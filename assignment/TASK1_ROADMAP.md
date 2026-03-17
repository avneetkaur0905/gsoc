# Task 1 Roadmap

## What Task 1 is asking

Build a model for painting classification using the ArtGAN / WikiArt data.

You need to:
- classify attributes like `Style`, `Genre`, and `Artist`
- use a normal baseline model first
- also build a **convolutional-recurrent** model
- evaluate the models properly
- inspect mistakes / outliers
- explain your strategy and results

In simple words:
- first build a model that works
- then build the assignment-style CNN + RNN model
- then compare them
- then explain what you learned

## The main steps of Task 1

### Step 1: Prepare the data

What this means:
- load the split files from ArtGAN
- connect them to the real WikiArt images
- make sure images open correctly

Status:
- done for `Style`

### Step 2: Build a small working subset

What this means:
- use a smaller sample of the data first
- this makes training possible on CPU

Status:
- done for `Style`

### Step 3: Build the baseline model

What this means:
- use a normal CNN classifier first
- in our case: `ResNet18`

Why:
- this gives a reference point
- later we compare CNN + RNN against this

Status:
- done for `Style`

### Step 4: Train and evaluate the baseline

What this means:
- run training for a few epochs
- track training and validation performance

Important metrics:
- training loss
- validation loss
- training accuracy
- validation accuracy

Status:
- done for `Style`

### Step 5: Build the convolutional-recurrent model

What this means:
- CNN extracts visual features
- RNN reads those features as a sequence
- in our case: `ResNet18 conv backbone + GRU`

Why:
- this is the part that directly matches the assignment requirement

Status:
- done for `Style`

### Step 6: Train and evaluate the CNN + RNN model

What this means:
- train the CNN + RNN model
- compare validation performance with the baseline

Status:
- done for `Style`

### Step 7: Compare the two models

What this means:
- which model trained better?
- which model validated better?
- did the CNN + RNN actually help?

Status:
- partly done for `Style`

### Step 8: Error analysis / outlier analysis

What this means:
- inspect wrong predictions
- find paintings that do not fit their assigned class well
- discuss ambiguous or unusual paintings

Status:
- notebook cells are prepared
- not fully finished yet

### Step 9: Repeat for `Genre`

What this means:
- change the notebook attribute from `Style` to `Genre`
- run the same pipeline

Status:
- not started

### Step 10: Repeat for `Artist`

What this means:
- change the notebook attribute from `Style` to `Artist`
- run the same pipeline

Status:
- not started

## What we have already completed

For `Style`, we have already done:
- data loading
- image path fixing
- image display check
- small subset creation
- safe dataset / dataloader
- baseline `ResNet18`
- baseline training
- CNN + RNN model
- CNN + RNN training

So the hard setup work is already done.

## Where we are right now

We are in the middle of `Style`.

More specifically:
- the `Style` pipeline works
- both models are built
- both models have been trained
- now we need to cleanly compare them and finish the analysis

## Current understanding of the results

### Baseline model

The baseline learned the training data strongly, but validation accuracy stayed much lower.

Meaning:
- baseline works
- some overfitting is happening

### CNN + RNN model

The CNN + RNN model also works.
In some runs it had weak validation performance.
In a later run, validation behavior improved more steadily.

Meaning:
- the model is valid
- the architecture is implemented
- but it still needs careful interpretation

## What is left to finish `Style`

These are the remaining steps for `Style`:

1. Save the best baseline outputs cleanly
2. Save the best CNN + RNN outputs cleanly
3. Create a small comparison table
4. Plot training / validation curves
5. Run confusion matrix
6. Inspect a few wrong predictions
7. Write a short interpretation section

After that, `Style` will be in good shape.

## What comes after `Style`

After `Style`, do the same workflow for:
- `Genre`
- `Artist`

That means:
- switch `ATTRIBUTE = "Genre"`
- rerun the notebook
- save outputs

Then:
- switch `ATTRIBUTE = "Artist"`
- rerun the notebook
- save outputs

## Recommended next plan

### Immediate next step

Finish `Style` cleanly:
- comparison table
- confusion matrix
- outlier examples
- short interpretation

### After that

Run `Genre` through the same notebook.

### After that

Run `Artist` through the same notebook.

## Rough progress estimate

For Task 1 overall:
- around 60% done

Breakdown:
- `Style`: mostly done, but needs analysis and cleanup
- `Genre`: not started
- `Artist`: not started

## Very short version

What Task 1 needs:
- load data
- train baseline
- train CNN + RNN
- compare them
- inspect mistakes
- repeat for Style, Genre, Artist

Where we are:
- `Style` is working
- both models are already built and trained

What to do next:
- finish `Style` analysis first
- then run `Genre`
- then run `Artist`
