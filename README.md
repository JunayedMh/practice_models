# practice_models

A curated collection of machine learning practice projects covering classification, regression, time series, and exploratory modeling. Each folder contains a Jupyter notebook (and in one case a Python script) with datasets and analysis steps used to build, evaluate, and visualize models.

## Repository Structure

- `Football match predictor/`
  - `footbal-match-prediction-model.ipynb` - Notebook for predicting football match outcomes.
  - `footbal-match-prediction-model.py` - Python script version of the model.
  - `former_names.csv`, `goalscorers.csv`, `results.csv`, `shootouts.csv` - Supporting football match datasets.

- `Heart Desease/`
  - `heart_disease.ipynb` - Notebook for heart disease prediction using medical dataset.
  - `heart.csv` - Heart disease dataset.

- `House Price model/`
  - `house_price.ipynb` - Notebook for house price prediction.
  - `house_price_train.csv` - Training data for the house price model.

- `IPL/`
  - `IPL.ipynb` - Notebook covering IPL match and player analytics.
  - `Players.xlsx` - Player details dataset.
  - `deliveries.csv`, `matches.csv`, `most_runs_average_strikerate.csv`, `teams.csv`, `teamwise_home_and_away.csv` - IPL datasets.

- `Netflix Title/`
  - `netflix.ipynb` - Notebook exploring Netflix titles and metadata.
  - `netflix_titles.csv` - Netflix dataset.

- `Spam Message Detection/`
  - `spam_detection.ipynb` - Notebook for spam detection model using text classification.
  - `spam.csv` - Spam message dataset.

## Getting Started

### Prerequisites

- Python 3.8+
- `pip`
- Jupyter Notebook or Jupyter Lab

### Install Dependencies

```bash
cd /workspaces/practice_models
pip install -r requirements.txt
```

### Launch Notebooks

```bash
jupyter notebook
```

Then open the notebook for the project you want to explore.

## Project Highlights

- **Football match predictor**: Predicts football match outcomes from match and historical result data.
- **Heart Desease**: Builds a classification model to detect heart disease from medical features.
- **House Price model**: Trains a regression model to estimate house prices from real estate features.
- **IPL**: Performs exploratory data analysis on IPL matches, players, and team statistics.
- **Netflix Title**: Analyzes Netflix catalog data for trends in titles, genres, and release years.
- **Spam Message Detection**: Uses NLP and classification techniques to distinguish spam from non-spam messages.

## Notes

- Folder names include spaces; use quotes when referencing them in terminal commands.
- The notebooks are self-contained and usually include dataset loading, preprocessing, modeling, and evaluation steps.
- In this repo I will be posting and updating my future models.

## License

This repository is provided for practice and learning purposes. Feel free to explore, modify, and expand the notebooks.
