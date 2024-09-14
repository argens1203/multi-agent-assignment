import torch
import copy
from constants import device, state_size
import numpy as np

# from core import state_size

statespace_size = state_size


# The function "prepare_torch" needs to be called once and only once at the start of your program to initialise PyTorch and generate the two Q-networks. It returns the target model (for testing).
def prepare_torch():
    global statespace_size
    global model, model_hat
    global optimizer
    global loss_fn
    l1 = statespace_size
    l2 = 150
    l3 = 100
    l4 = 4
    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4),
    ).to(device)
    model_hat = copy.deepcopy(model).to(device)
    model_hat.load_state_dict(model.state_dict())
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model_hat


# The function "update_target" copies the state of the prediction network to the target network. You need to use this in regular intervals.
def update_target():
    global model, model_hat
    model_hat.load_state_dict(model.state_dict())


# The function "get_qvals" returns a numpy list of qvals for the state given by the argument _based on the prediction network_.
def get_qvals(state):
    return model(state).to(device)


# The function "get_maxQ" returns the maximum q-value for the state given by the argument _based on the target network_.
def get_maxQ(s):
    maxxed = torch.max(model_hat(s), dim=1)
    return torch.max(model_hat(s), dim=1).values.float()


# The function "train_one_step_new" performs a single training step. It returns the current loss (only needed for debugging purposes). Its parameters are three parallel lists: a minibatch of states, a minibatch of actions, a minibatch of the corresponding TD targets and the discount factor.
def train_one_step(states, actions, targets):
    # print(states)
    # print(states.shape)
    # for s in states:
    # print(s.shape)
    # pass to this function: state1_batch, action_batch, TD_batch
    global model, model_hat
    # state1_batch = torch.cat([torch.from_numpy(s).float() for s in states])
    state1_batch = states.to(device)
    action_batch = actions.to(device)
    # print(action_batch.shape)
    # print(state1_batch.shape)
    Q1 = model(state1_batch)
    X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
    Y = targets.clone().detach().to(device).float()
    loss = loss_fn(X, Y)
    # print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item() / len(X)
