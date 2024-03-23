import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.padqn import PADQNAgent
from agents.utils import hard_update_target_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiPassQActor(nn.Module):
    """
    Class implementation of QActor with Multiple Passes.
    """
    def __init__(
            self,
            state_size,
            action_size,
            action_parameter_size,
            activation="relu",
            **kwargs
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        assert (activation in {"relu", "leaky_relu"}), "invalid activation."
        self.activation = activation

        # Creating the neural-network architecture
        self.layers = nn.ModuleList()
        input_size = self.state_size + self.action_size
        hidden_layers = [128]
        nh = len(hidden_layers)
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        for i in range(1, nh):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        last_hidden_layer_size = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(last_hidden_layer_size, self.action_size))

        # Initializing the neural-network
        # initialize hidden layers
        for i in range(len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=self.activation)
            nn.init.zeros_(self.layers[i].bias)
        # initialize output layer
        nn.init.normal_(self.layers[-1].weight, mean=0., std=1.)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state, action_parameters):
        """
        :purpose: implement forward pass.
        :param state: {Tensor(batch_size, 9)}
        :param action_parameters: {Tensor(batch_size, 3)}
        :return: Q {Tensor(batch_size, 3)}
        """
        negative_slope = 0.01

        Q = []
        # first, we duplicate inputs, so we can process all actions in a single pass
        batch_size = state.shape[0]
        x = torch.cat((state, torch.zeros_like(action_parameters)), dim=1)  # {Tensor(batch_size, 12)}
        x = x.repeat(self.action_size, 1)  # x is Tensor(3 * batch_size, 12)
        for a in range(self.action_size):
            x[a * batch_size:(a + 1) * batch_size, self.state_size + a: self.state_size + (a + 1)] \
                = action_parameters[:, a: (a + 1)]

        # pass through the hidden layers
        for i in range(0, len(self.layers) - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        # pass through the output layer
        Q_all = self.layers[-1](x)  # {Tensor(3 * batch_size, 12)}

        # extract Q-values for each action
        for a in range(self.action_size):
            Q_a = Q_all[a * batch_size:(a + 1) * batch_size, a]  # {Tensor(batch_size,)}
            Q_a = Q_a.unsqueeze(-1)  # {Tensor(batch_size, 1)}
            Q.append(Q_a)
        Q = torch.cat(Q, dim=1)  # {Tensor(batch_size, 3)}
        return Q


class MultiPassPADQNAgent(PADQNAgent):
    """
    Class implementation of MultiPassPADQNAgent
    """
    NAME = "Multi-Pass PA-DQN Agent"

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'MPDQN'
        # instantiating the actor
        self.actor = MultiPassQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_size,
                                     ).to(device)
        # instantiate the target for actor
        self.actor_target = MultiPassQActor(self.observation_space.shape[0],
                                            self.num_actions,
                                            self.action_parameter_size).to(device)
        # copy parameters of actor to its target
        hard_update_target_network(self.actor, self.actor_target)
        # set the target in eval mode.
        self.actor_target.eval()
        # initialize optimizer
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.alpha_actor)
