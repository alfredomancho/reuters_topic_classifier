## Reuters Topic Classifier

## Overview
A lightweight text-classification project using the Reuters newswire dataset, with a focus on model calibration to produce reliable confidence estimates. It demonstrates: 

- Loading and preprocessing the Reuters dataset
- Building and training a neural model (Embedding + CNN)
- Applying regularization (e.g. label smoothing)
- Calibrating the model’s predicted probabilities via:
  - Label smoothing (during training)
  - Temperature scaling (post-training)
  - Isotonic regression (post-training)
- Evaluating calibration using Expected Calibration Error (ECE) and maximum deviation metrics

## Structure
- **model.py**: Complete script for data prep, model creation, training, evaluation, plotting and calibration.  Saves trained model to .keras file.
- **loss_curves.png**: Sample plot showing training and validation loss over epochs.
- **cal_curve.png**: Sample baseline reliability plot.
- **cal_curve_tempCal.png**: Sample reliability plot after temperature scaling and quantile binning applied.
- **cal_curve_tempCal_iso.png**: Sample reliability plot after temperature scaling, quantile binning and isotonic regression on test set applied.
- **requirements.txt**: List of required packages and dependencies.

## Calibration steps 
- Label-smoothing (0.05)
- Temperature scaling:  Scalar T was optimized on validation set
- Isotonic regression

## Results
+ Baseline ECE: 0.0693, Max Deviation: 0.2486
+ Temperature scaling & quantile binning ECE: 0.0308, Max Deviation: 0.0919
+ Isotonic regression on test set ECE: 0.0201, Max Deviation: 0.0771

- The above metrics, especially maximum deviation, varied by about +/- 4-5% every time the model was retrained.  Re-run with --seed and larger calibration set to stabilize metrics.

## How to Run
1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt

