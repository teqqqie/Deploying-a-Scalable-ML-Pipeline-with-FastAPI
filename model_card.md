# Model Card

## Model Details

This model was developed by Caleb Poock. The model used is a scikit-learn RandomForestClassifier instance with default parameters.

Full project repository located at https://github.com/teqqqie/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.

## Intended Use

This model uses a collection of demographic data to predict whether a person makes more or less than $50k a year.

Demographics used:
- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours Worked Per Week
- Native Country

## Training Data

This model was trained on the Census Income dataset published by the UC Irving Machine Learning Repository, found here: https://archive.ics.uci.edu/dataset/20/census+income.

## Evaluation Data

The model was evaluated using a test split of 20% of the original dataset.

## Metrics

Overall metrics:
- Precision: 0.7519
- Recall: 0.6309 
- F1: 0.6861

Metrics for specific slices of the data can be found in `slice_output.txt` in the project repository.

## Ethical Considerations

This model was trained and tested using census data. Some users may perceive the model as discriminatory based on certain demographic information used.

## Caveats and Recommendations

This model was trained on data from 1994, and may not accurately reflect the significant economic changes that have occurred since then.