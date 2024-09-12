import matplotlib.pyplot as plt

from frontend import Visualization, Controller, Model

if __name__ == "__main__":
    game = Model()
    controller = Controller(game, 1000)

    fig1, ax1 = plt.subplots()
    vis = Visualization(game, controller, fig1, ax1)

    # game = Game()
    # controller = Controller(game, 1000)
    # controller.train(1000)
    # fig1, ax1 = plt.subplots()
    # vis = Visualization(game, controller, fig1, ax1)
    # Visualization.plot_training(controller.get_metrics())
