
- **Conformal Prediction (CP)** is a method for getting calibrated uncertainty estimates for machine learning models where they are not generally available, something like confidence intervals for almost any model.

- Surprisingly, it works for classification problems as well as regression.
    - The regression case produces a prediction interval, similar to a confidence interval.
    - The classification case produces sets of predictions that are guaranteed to contain the true class label with a certain probability.

- CP works by estimating a "non-conformity" score for each sample and comparing that to a distribution of non-conformity (NC) scores derived from a subset of the data.
   - The data are split into test, train, and calibration sets.
   - NC scores are calculated on the calibration set, and the 1-alpha quantile of these scores ($\hat q$) is noted.
   - When new data is encountered, the NC of the new data is compared to predicted probabilities for each class.
   - Potential outcomes are included in the final prediction set if the NC score is less than or equal to $\hat q$.
   - In the regression case, $\hat q$ is derived from the residuals and used to create prediction intervals by adding and subtracting $\hat q$ from the predicted value.
