from q-learning-nim import QAgent, Nim, Opponent, playQvOpp


PROPORTION_RATE = 0.1

if __name__ == '__main__':
    game = Nim(10)
    qagent = QAgent(game, 1.0, 0.45, 0.1)
    opponent = Opponent(game, 'rand')

    playQvOpp(100, game, qagent, opponent, PROPORTION_RATE):

