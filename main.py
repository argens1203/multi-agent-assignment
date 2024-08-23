import matplotlib.pyplot as plt
from agent import Agent
from grid import GridWorld as Environment
from visualization import Visualization


def plot_training(results):
    iterations = [t[0] for t in results]
    losses = [t[3] for t in results]
    total_rewards = [t[2] for t in results]

    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting the loss in the first subplot
    ax1.plot(iterations, losses, marker="o", label="Loss")
    ax1.set_title("Iteration vs Loss")
    ax1.set_xlabel("Iteration Number")
    ax1.set_ylabel("Loss")

    # Plotting the total rewards in the second subplot
    ax2.plot(
        iterations, total_rewards, label="Total Reward", color="orange", marker="o"
    )
    ax2.set_title("Iteration vs Total Reward")
    ax2.set_xlabel("Iteration Number")
    ax2.set_ylabel("Total Reward")

    # Display the plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = Environment(size=8)
    agent = Agent(env)
    training_record = agent.train(50000)
    plot_training(training_record)
    visualization = Visualization(env, agent)

    # ani = FuncAnimation(visualization.fig, visualization.animate, frames=400, interval=400, repeat=False)
    plt.show()
