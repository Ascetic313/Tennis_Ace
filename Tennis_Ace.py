
#Library:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data:
data = pd.read_csv("tennis_stats.csv")
df = pd.DataFrame(data=data)
df.head()

#Plotting graphs:

plt.scatter(df[['FirstServePointsWon']], df[["Winnings"]])
plt.xlabel('FirstServePointsWon')
plt.ylabel('Winnings')
plt.show()
plt.scatter(df[['BreakPointsOpportunities']], df[["Winnings"]])
plt.xlabel(xlabel='BreakPointsOpportunities')
plt.ylabel(ylabel='Winnings')
plt.show()
plt.scatter(df[['TotalPointsWon']], df[["Winnings"]])
plt.xlabel(xlabel='TotalPointsWon')
plt.ylabel(ylabel='Winnings')
plt.show()

# define x axis							
x = df[['FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon',
        'SecondServePointsWon', 'SecondServeReturnPointsWon', 'Aces',
        'BreakPointsConverted', 'BreakPointsFaced', 'BreakPointsOpportunities',
        'BreakPointsSaved', 'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon',
        'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalPointsWon',
        'TotalServicePointsWon', 'Wins', 'Losses', 'Ranking']]
# define y axis
y = df[['Winnings']]
mlr = LinearRegression()
mlr.fit(x_train, y_train)
y_predict = mlr.predict(x_test)
plt.scatter(y_test,y_predict)
plt.xlabel("Predicted Winnings")
plt.ylabel("Actual Winnings")
plt.show()


print("Accuracy on training set = ", mlr.score(x_train, y_train))
print("Accuracy on testing set = ", mlr.score(x_test, y_test))
print("model coff =", mlr.coef_)

# redefiny x axises
x = df[['FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon',
        'SecondServePointsWon', 'SecondServeReturnPointsWon', 'BreakPointsConverted',
        'BreakPointsSaved', 'ReturnGamesWon', 'ReturnPointsWon', 'ServiceGamesPlayed',
        'ServiceGamesWon', 'TotalPointsWon', 'TotalServicePointsWon', 'Wins', 'Losses']]

y = df[['Winnings']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
mlr.fit(x_train, y_train)
y_predict = mlr.predict(x_test)
print("Accuracy on training set = ", mlr.score(x_train, y_train))
print("Accuracy on testing set = ", mlr.score(x_test, y_test))