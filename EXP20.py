import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5', 'Player6', 'Player7', 'Player8', 'Player9', 'Player10'],
    'Age': [25, 28, 22, 30, 24, 26, 29, 31, 27, 23],
    'Position': ['Forward', 'Midfielder', 'Forward', 'Defender', 'Midfielder', 'Forward', 'Defender', 'Midfielder', 'Forward', 'Midfielder'],
    'GoalsScored': [20, 15, 25, 10, 18, 22, 8, 14, 21, 16],
    'WeeklySalary': [5000, 6000, 4500, 7000, 5500, 7500, 8000, 6500, 7000, 6000]
}

df = pd.DataFrame(data)
df.to_csv('soccer_players.csv', index=False)

df = pd.read_csv('soccer_players.csv')

top_goals_players = df.nlargest(5, 'GoalsScored')

top_salary_players = df.nlargest(5, 'WeeklySalary')

average_age = df['Age'].mean()

above_average_age_players = df[df['Age'] > average_age]['Name']

position_distribution = df['Position'].value_counts()
position_distribution.plot(kind='bar', color='cyan')
plt.title('Distribution of Players Based on Positions')
plt.xlabel('Position')
plt.ylabel('Number of Players')
plt.show()

print("Top 5 Players with the Highest Number of Goals:")
print(top_goals_players[['Name', 'GoalsScored']])
print("\nTop 5 Players with the Highest Salaries:")
print(top_salary_players[['Name', 'WeeklySalary']])
print(f"\nAverage Age of Players: {average_age:.2f}")
print("\nPlayers Above the Average Age:")
print(above_average_age_players)
