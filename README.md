# Vacancy Forecasting & Simulation - Rental Portfolio

## Problem
Rental portfolio vacancy and lease-renewal risk were being estimated by gut feel, 
with no probabilistic view of downtime or which submarkets carried the most risk.

## Approach
- Submarket clustering to group units by comparable risk profile
- XGBoost models predicting renewal probability and expected vacancy downtime per unit/cluster
- Simulation layer generating a probability distribution over portfolio-level vacancy outcomes, not a single point estimate
- Served via a lightweight API so predictions and simulated scenarios are queryable on demand, not locked in a notebook

## Evolution from earlier work
Builds on an earlier Monte Carlo simulation prototype by adding a trained 
predictive layer in front of the simulation, and moving from a standalone app to an API.

## Status
Active — predictive + API layer in place; data/pipeline stages already built by team

## Limitations
[State plainly, e.g.: trained on a single time snapshot (Nov '22 clustering); would need retraining/drift monitoring for production use.]