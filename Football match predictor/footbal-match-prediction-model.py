import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('/workspaces/practice_models/files/football/results.csv')
print(df.head())
print(df.columns.tolist())

df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
print(df['home_win'])

df['away_win'] = (df['away_score'] > df['home_score']).astype(int)
print(df['away_win'])

df['draw'] = (df['home_score'] == df['away_score']).astype(int)
print(df['draw'])

df['goal_difference'] = df['home_score'] - df['away_score']
print(df['goal_difference'])

home_counts = df.groupby('home_team').size().reset_index(name='home_games_played')
df = df.merge(home_counts, on='home_team', how='left')

away_counts = df.groupby('away_team').size().reset_index(name='away_games_played')
df = df.merge(away_counts, on='away_team', how='left')

home_avg = df.groupby('home_team')['home_score'].mean().reset_index()
home_avg.columns = ['home_team', 'home_avg_goals']
df = df.merge(home_avg, on='home_team', how='left')
print(df[['home_team', 'home_win', 'home_avg_goals']].head(10))

away_avg = df.groupby('away_team')['away_score'].mean().reset_index()
away_avg.columns = ['away_team', 'away_avg_goals']
df = df.merge(away_avg, on='away_team', how='left')
print(df[['away_team', 'away_win', 'away_avg_goals']].head(10))

tournaments_type = df['tournament'].unique()
print(tournaments_type)
print(f'Number of unique tournaments: {len(tournaments_type)}')

neutral_check = df['neutral'].unique()
print(neutral_check)
print(f'Number of neutrality: {len(neutral_check)}')

features = ['home_avg_goals', 'home_games_played', 'away_avg_goals', 'away_games_played', 'draw', 'neutral']
X = df[features].dropna()
Y = df.loc[X.index, 'home_win']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=60)

model = LogisticRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(Y_test, Y_pred):.2%}')

features = ['home_avg_goals', 'home_games_played', 'away_avg_goals', 'away_games_played', 'draw', 'neutral']
X = df[features].dropna()
Y = df.loc[X.index, 'home_win']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=60)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)

Y_pred = rf_model.predict(X_test)
print(f'Accuracy: {accuracy_score(Y_test, Y_pred):.2%}')

for feature, coef in zip(features, model.coef_[0]):
  print(f'{feature}: {coef:.3f}')

england_home_avg_goals = df[df['home_team'] == 'England']['home_avg_goals'].iloc[0]
england_home_games_played = df[df['home_team'] == 'England']['home_games_played'].iloc[0]

brazil_away_avg_goals = df[df['away_team'] == 'Brazil']['away_avg_goals'].iloc[0]
brazil_away_games_played = df[df['away_team'] == 'Brazil']['away_games_played'].iloc[0]

prediction_features = {
    'home_avg_goals': [england_home_avg_goals],
    'home_games_played': [england_home_games_played],
    'away_avg_goals': [brazil_away_avg_goals],
    'away_games_played': [brazil_away_games_played],
    'draw': [0],
    'neutral': [0]
}

prediction_df = pd.DataFrame(prediction_features)
predicted_outcome = model.predict(prediction_df)

if predicted_outcome[0] == 1:
    print("Prediction: England (Home Team) is predicted to win.")
else:
    print("Prediction: England (Home Team) is NOT predicted to win (could be a draw or Brazil win).")

