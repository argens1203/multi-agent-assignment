import matplotlib.pyplot as plt
from .grid import GridWorld as Environment
from .agent import Agent
from .visualization import Visualization

debug = False


class Game:
    def run(self):
        self.env = Environment(size=8)
        self.agent = Agent(self.env)
        training_record = self.train_agent(50000)
        plot_training(training_record)
        Visualization(self.env, self.agent).show()

    def train_agent(self, episodes):
        training_record = []
        for _ in range(episodes):
            self.env.reset()
            state = self.env.get_state()
            max_reward = self.env.calculate_max_reward()

            total_reward = 0
            while not self.env.is_terminal():
                action = self.agent.choose_action(state)
                next_state, reward, terminal = self.env.move(action)
                self.agent.learn(state, action, reward, next_state, terminal)

                state = next_state
                total_reward += reward

            loss = max_reward - total_reward
            if debug:
                print(
                    f"Episode {_} completed with total reward: {total_reward},max_reward:{max_reward}, loss:{loss}"
                )
            training_record.append([_, max_reward, total_reward, loss])

        print("Training complete")
        return training_record


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
