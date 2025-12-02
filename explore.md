# Data Exploration

This document contains notes and insights from exploratory data analysis.

## Dataset Statistics

### Training Set
- Total images: (to be filled after running analysis)
- Healthy samples: 
- Melanoma samples:
- Class balance:

### Validation Set
- Total images:
- Healthy samples:
- Melanoma samples:
- Class balance:

## Image Analysis

### Size Distribution
- Most common image sizes:
- Aspect ratios:

### Color Distribution
- Mean RGB values:
- Standard deviation:

## Class Imbalance

The dataset shows class imbalance that needs to be addressed during training:
- Strategy 1: Use pos_weight in BCEWithLogitsLoss
- Strategy 2: Use balanced accuracy as evaluation metric

## Visualizations

See `notebooks/data-exploring.ipynb` for detailed visualizations.

## Next Steps

1. Apply data augmentation to training set
2. Implement balanced sampling
3. Monitor balanced accuracy during training
4. Consider ensemble methods if single model performance plateaus
