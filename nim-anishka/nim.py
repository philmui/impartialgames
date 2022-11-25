import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

GAME_SIZE = 15
ACTION_SPACE = [i for i in range(1, 5)]
Q_WINS = []


class Game:
    def __init__(self):
        self.gameList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.done = False

    def get_game_list(self):
        return (self.gameList)

    def opponent_move(self):
        randomAction = random.randint(1, len(ACTION_SPACE))
        del self.gameList[0:randomAction]
        return (randomAction)

    def qAgent_move(self, qTable, oldState):
        index = oldState - 1
        if max(qTable[index])  <= 0.5:
            QAction = random.randint(1, len(ACTION_SPACE))
        else:
            QAction = (qTable[index].index(max(qTable[index])) + 1)
            QAction = int(round(QAction,0))
        del self.gameList[0:QAction]
        return (QAction)

    def get_state(self):
        return (len(self.gameList))

    def game_over(self, state):
        #state = get_state()
        #print("gameover: ", str(state))
        if state == 0:
            return (True)
        else:
            return (False)

class Q_class:
    def __init__(self):
        self.reward = 0
        self.qTable = []
        for i in range(GAME_SIZE):
            self.qTable.append([0] * len(ACTION_SPACE))

    def get_QTable(self):
        return (self.qTable)

    def update_QTable(self, oldState, newState, action, reward):
        if (reward !=1) and (reward !=-1):
            updateReward = round(reward + 0.2*(self.best_future_reward(newState)), 5)
            self.qTable[oldState-1][action-1] += updateReward
        else:
            self.qTable[oldState-1][action-1] = reward

    def best_future_reward(self, newState):
        return (max(self.qTable[newState]))


def trainAI(numTrials):
    qClass = Q_class()
    qTable = qClass.get_QTable()
    opponentWin = 0
    qWin = 0

    for i in range (numTrials):
        game = Game()
        print ("Game # .....", str(i))
        while True:
            oldState = game.get_state()
            action = game.qAgent_move(qTable, oldState)
            newState = game.get_state()
            if game.game_over(newState) == True:
                print ("Opponent won")
                Q_WINS.append(0)
                qClass.update_QTable(oldState, newState, action, -1)
                opponentWin+=1
                break
            else:
                qClass.update_QTable(oldState, newState, action, 0)

            oldState = game.get_state()
            action = game.opponent_move()
            newState = game.get_state()
            if game.game_over(newState) == True:
                print ("Q-agent won")
                Q_WINS.append(1)
                qClass.update_QTable(oldState, newState, oldState-1, 1) #update remaining states -1
                qWin += 1
                break

    print ("Total QWin: ", qWin, " Total Opponent Win: ", opponentWin)


def graph():
    GRAPH_SMOOTHING = 5

    q_win_x = []
    q_win_y = []
    for x in range(GRAPH_SMOOTHING, len(Q_WINS)):
        values_to_average = Q_WINS[max(0, x - GRAPH_SMOOTHING):x + 1]
        rolling_average = float(sum(values_to_average)) / len(values_to_average)
        q_win_y.append(rolling_average)
        q_win_x.append(x)
    plt.plot(q_win_x, q_win_y, label="Q-Learner Win Rate")
    plt.fill_between(q_win_x, 0, q_win_y, label="Q-Learner Win Rate")
    plt.title("Win Rate  " + str(GRAPH_SMOOTHING) + " Games)")
    plt.savefig("result.png")



if __name__ == "__main__":
    trainAI(5000)
    graph()



