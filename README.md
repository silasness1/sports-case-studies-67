# General

Sports data part of Case studies WS26/27: 
Nick, Rohit, Sriram, Silas


# File Organization

- File paths in scripts are relative to the project root (e.g. `"raw-data/EuroTest.csv"`), so open `sports-case-studies-67.Rproj` in RStudio or run scripts from the top-level folder. Please avoid absolute paths like `C:/Users/...` so the scripts run for everyone

## Folders

- docs: contains explanations from course materials like exercise sheet 1 for the first week
    - `getting-on-the-same-page-about-splits.txt` explains the leave-one-tournament-out / nested CV splits we agreed on

- raw-data: all input data
    - football: `EuroTraining.csv` / `EuroTest.csv` (original split), `euro_<year>.csv` (one file per tournament, made by `football-wk3-train-test-split.R`), `*_with_age.csv` and `Eurodata_with_Age_New.csv` (squad age covariates incl. AgeFitDiff)
    - handball: `handball_long.csv`, `handball_wide.csv`
    - tennis: `SlamsData.csv`

- weeks-1-2: scripts from the first two weeks, split into `week-1` and `week-2` subfolders (data exploration, first trees for all 3 sports)

- combined: where the final code lives. The idea is one script that builds the data objects and model fits, and separate scripts that only read those fits to make results and plots
    - `football_combined_nested_tune.R`: fits all football models (null, poisson, random forest, xgboost, nnet) with and without age covariates in one nested CV and saves them to `results/combined/*.rds`. Sourcing it can refit and overwrite those, so check its settings before running
    - `community_plots_result_pres.R`: shared results, tables and plots for the group presentation. It only sources the fitting script when every RERUN switch is FALSE, QUICK is FALSE and all fits are saved, and its `save_plot()` / `save_table()` write to `results/combined/plots` and `results/combined/tables`
    - individual scripts (planned): one per person for their own results and plots, e.g. `<name>_results.R`
    - result/plot scripts should load the saved `.rds` fits rather than sourcing the fitting script, so they don't trigger a refit

- results: saved model fits and plots
    - `combined`: output of the combined scripts (fits as `.rds`, plus `plots/` and `tables/`)
    - `Neural Networks`: nnet diagnostic plots
    - `_archive`: older outputs from `football-wk4-nestedtune.R`, including the frozen runs from 2026-11-11 used in the week 4 presentation

- NN-Mess: neural network experiments (direct classification vs. estimated goals, basic vs. extended predictors)

- xgboost_with_skellam: xgboost with Skellam-based win/draw/loss probabilities, plus its saved model and predictions

- literature: papers we've referenced, `splits` has the cross-validation ones

- presentations: please make a new subfolder for different presentations so it's cleaner when compiling latex
    - Note: See `links-to-google-slides-pres.txt` for links to presentations hosted in google slides

- renv: it's an environment managment package for R

## Scripts in the top-level folder

- `football-wk3-poisson.R`: first nested CV for penalized poisson regression (week 3)
- `football-wk3-train-test-split.R`: splits the football data into one csv per tournament
- `sports-week3.R`: notes on the tournament structure from the 11/1 meeting
- `football-wk4-nestedtune.R`: week 4 version with poisson, random forest and xgboost using parsnip/workflows/nestedtune
- `Random_Forest_*`: random forest with leave-one-tournament-out / nested CV and Skellam thresholds (`_Macro_F1` tunes the draw threshold on macro F1)
- `xgboost_week3.R`: xgboost with nested CV (week 3)
- `Case Studies(nnet and multilog reg).R`: neural network and multinomial logistic regression classification


# Other notes

- Anticipate making a new git repo after the sports project ends

- Feel free to create a `clean-data` folder or something similar if necessary
