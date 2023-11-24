import pandas as pd

data = {
    'Name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5', 'Player6', 'Player7', 'Player8', 'Player9', 'Player10'],
    'Age': [25, 28, 22, 30, 26, 24, 27, 29, 31, 23],
    'Position': ['Forward', 'Midfielder', 'Forward', 'Defender', 'Forward', 'Midfielder', 'Forward', 'Defender', 'Forward', 'Midfielder'],
    'GoalsScored': [15, 10, 18, 5, 20, 12, 22, 8, 25, 14],
    'WeeklySalary': [100000, 90000, 120000, 80000, 150000, 110000, 180000, 75000, 200000, 95000]
}

df = pd.DataFrame(data)

df.to_csv('soccer_players.csv', index=False)

import matplotlib.pyplot as plt

df = pd.read_csv('soccer_players.csv')

top_goals_players = df.nlargest(5, 'GoalsScored')
print("\nTop 5 Players with the Highest Number of Goals Scored:")
print(top_goals_players[['Name', 'GoalsScored']])

top_salary_players = df.nlargest(5, 'WeeklySalary')
print("\nTop 5 Players with the Highest Salaries:")
print(top_salary_players[['Name', 'WeeklySalary']])

average_age = df['Age'].mean()
print(f"\nAverage Age of Players: {average_age:.2f}")

above_average_age_players = df[df['Age'] > average_age]
print("\nPlayers Above the Average Age:")
print(above_average_age_players[['Name', 'Age']])

position_distribution = df['Position'].value_counts()
position_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribution of Players based on Positions')
plt.xlabel('Position')
plt.ylabel('Number of Players')
plt.show()
