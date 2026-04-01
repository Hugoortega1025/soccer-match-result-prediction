# Soccer Match Outcome Prediction

Predicting soccer match outcomes using in-game team performance statistics from the 2024-25 ESPN Soccer dataset. Built with Python and scikit-learn across 56,000+ matches.
## Research Question
Which in-game performance metrics best predict the outcome of a soccer match?

## Project Overview
This project builds and evaluates three machine learning classification models to predict whether a team wins, draws, or loses based on their in-game performance statistics. The analysis spans data cleaning, exploratory data analysis, feature engineering, modeling, and evaluation.


## Dataset
[ESPN Soccer Data — Kaggle](https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data)

- 30,000+ match fixtures from the 2024-25 season
- 400+ leagues worldwide
- Team-level performance statistics per match including possession, shots, passing, defensive actions, and disciplinary metrics


## Methodology
1. **Data Loading and Merging** — teamStats and fixtures joined on eventId
2. **Data Cleaning** — null removal, possession outlier filtering, division by zero handling
3. **Exploratory Data Analysis** — result distribution, possession vs shots scatter, Pearson correlation
4. **Feature Engineering** — three proxy metrics engineered based on soccer/football domain knowledge
5. **Classification Modeling** — Logistic Regression, Random Forest, Gradient Boosting
6. **Evaluation** — accuracy, classification report, confusion matrices

## Engineered Features

`shot_efficiency`: Ratio of shots on target to total shots — captures clinical finishing quality

`threat_per_possession`: Shots on target per possession % — captures attacking danger regardless of playing style

`ball_recovery`: Effective tackles + interceptions — captures defensive pressing activity

## Results
All three models converged at approximately 55% accuracy. No single model significantly outperformed the others, suggesting the limiting factor is the feature set rather than model complexity.

| Model | Accuracy | Win F1 | Loss F1 | Draw F1 |

| Logistic Regression | 55% | 0.64 | 0.61 | 0.02 |

| Random Forest | 54% | 0.63 | 0.60 | 0.13 |

| Gradient Boosting | 56% | 0.65 | 0.62 | 0.02 |

Draws were consistently the hardest outcome to predict across all models — a reflection of the high variance nature of the sport where two teams can produce nearly identical stat lines and still draw.

## Key Finding
`threat_per_possession` — was the single most important feature identified by the Random Forest. This validates the hypothesis that attacking lethality relative to ball possession is a stronger predictor of match outcome than raw possession or shot volume alone. Counter-attacking teams can win with low possession if they are clinical, something this metric captures directly.

## Limitations
- No event-level positional data — a true xG model requires shot location, angle, and defensive positioning
- Ball recovery does not account for field position — recovery in the final third carries significantly more tactical weight than recovery in a team's own half
- Weather, match importance, squad depth, and in-game events such as early red cards are not captured
- Dataset covers a single season limiting longitudinal trend analysis

## Tools
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Author
Hugo — MS Data Science, Pace University  
