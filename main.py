import matplotlib.pyplot as plt
from agent import Agent
from grid import GridWorld as Environment
from visualization import Visualization
from game import Game


if __name__ == "__main__":
    env = Environment(size=8)
    agent = Agent(env)
    training_record = agent.train(50000)
    Game.plot_training(training_record)
    visualization = Visualization(env, agent)

    # ani = FuncAnimation(visualization.fig, visualization.animate, frames=400, interval=400, repeat=False)
    plt.show()
