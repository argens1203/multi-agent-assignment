import matplotlib.pyplot as plt

from frontend import Visualization, Controller, Model, Storage, Trainer

if __name__ == "__main__":
    model = Model()

    max_itr = 1000
    storage = Storage(max_itr)
    trainer = Trainer(max_itr)
    controller = Controller()

    fig1, ax1 = plt.subplots()
    vis = Visualization(fig1, ax1)

    trainer.bind(model, storage)

    controller.bind(model, storage, trainer)
    vis.bind(model, controller).show()

    # game = Game()
    # controller = Controller(game, 1000)
    # controller.train(1000)
    # fig1, ax1 = plt.subplots()
    # vis = Visualization(game, controller, fig1, ax1)
    # Visualization.plot_training(controller.get_metrics())
