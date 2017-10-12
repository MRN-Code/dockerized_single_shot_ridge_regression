# dockerized_single_shot_ridge_regression
This repository save the dockerized single shot ridge regression for COINSTAC platform

Computation Phase 01: @local
Inputs:
covariates (X), dependents(y), lambda

Outputs:
beta, r_squared, t-value, p-value, mean(y), len(y)

Computation Phase 02: @remote
Inputs:
beta, r_squared, t-value, p-value, Local Degrees of Freedom
mean(y_local), len(y_local)

Outputs:
average(beta), Global Mean of y, Global Degrees of Freedom

Computation Phase 03: @local
Inputs:
average(beta), Global Mean of y

Outputs:
SSE_local, SST_local, Variance-Covariance Matrix

Computation Phase 04: @remote
Inputs:
SSE_local, SST_local, variance-covariance matrix, len(y)

Outputs:
Global R-squared, Global t-statistic vector