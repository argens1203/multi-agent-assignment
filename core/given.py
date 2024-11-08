import torch
import copy
from constants import device, state_size
import numpy as np

# from core import state_size


class DQN:
    def __init__(self, state_size, action_size=4):
        l1 = state_size
        l2 = 250
        l3 = 250
        l5 = action_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3, l5),
        ).to(device)

        self.model2 = copy.deepcopy(self.model).to(device)
        self.model2.load_state_dict(self.model.state_dict())
        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    # The function "update_target" copies the state of the prediction network to the target network. You need to use this in regular intervals.
    def update_target(self):
        self.model2.load_state_dict(self.model.state_dict())

    # The function "get_qvals" returns a numpy list of qvals for the state given by the argument based on the prediction network.
    def get_qvals(self, state):
        q_values = self.model(state.to(device)).to(device)
        return q_values

    # The function "get_maxQ" returns the maximum q-value for the state given by the argument based on the target network.
    def get_maxQ(self, state):
        return torch.max(self.model2(state.to(device)), dim=1).values.float()

    # The function "train_one_step_new" performs a single training step.
    # It returns the current loss (only needed for debugging purposes).
    # Its parameters are three parallel lists: a minibatch of states, a minibatch of actions,
    # a minibatch of the corresponding TD targets and the discount factor.
    def train_one_step(self, states, actions, targets):
        # state1_batch = torch.cat([torch.from_numpy(s).float() for s in states])
        state1_batch = states.to(device)
        action_batch = actions.to(device)
        targets = targets.to(device)
        # print(action_batch.shape)
        # print(state1_batch.shape)
        Q1 = self.model(state1_batch)
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        Y = targets.clone().detach().to(device).float()
        loss = self.loss_fn(X, Y)
        # print(loss)
        self.optimizer.zero_grad()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5000)
        loss.backward()

        # total_norm = 0
        # for p in self.model.parameters():
        #     param_norm = p.grad.data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1.0 / 2)
        # print(total_norm)

        self.optimizer.step()
        return loss.item() / len(X)

    def save(self, prefix):
        torch.save(self.model.state_dict(), f"{prefix}_1.pth")
        torch.save(self.model2.state_dict(), f"{prefix}_2.pth")

    def load(self, prefix):
        try:
            self.model.load_state_dict(
                torch.load(f"{prefix}_1.pth", weights_only=False, map_location=device)
            )
            self.model2.load_state_dict(
                torch.load(f"{prefix}_2.pth", weights_only=False, map_location=device)
            )
        except:
            print("load state_dict failed")
