import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from agents.agent import Agent
from agents.memory.memory import Memory
from agents.utils import soft_update_target_network, hard_update_target_network
from agents.utils.noise import OrnsteinUhlenbeckActionNoise


# %-----------------------------------------------------------------------------------%
# %------------------------ CLASS IMPLEMENTATION OF QActor ---------------------------%
# %-----------------------------------------------------------------------------------%
class QActor(nn.Module):
    def __init__(
            self,
            state_size,
            action_size,
            action_parameter_size,
            activation="leaky_relu",
            **kwargs
    ):
        super(QActor, self).__init__()
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
        negative_slope = 0.01  # slope for leaky_relu
        x = torch.cat((state, action_parameters), dim=1)  # Tensor(batch_size, 12)
        # pass through the hidden layers
        for i in range(0, len(self.layers) - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        # pass through the output layer
        Q = self.layers[-1](x)  # Tensor(batch_size, 3)
        return Q


# %-----------------------------------------------------------------------------------%
# %---------------------- CLASS IMPLEMENTATION OF ParamActor -------------------------%
# %-----------------------------------------------------------------------------------%
class ParamActor(nn.Module):
    def __init__(
            self,
            state_size,
            action_size,
            action_parameter_size,
            activation="leaky_relu",
            **kwargs
    ):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        assert (activation in {"relu", "leaky_relu"}), "invalid activation."
        self.activation = activation

        # Creating neural-network architecture
        input_size = self.state_size
        self.num_experts = 4
        self.expert_output_dim = 32  # experts must have the same output dimension
        # gates, one for each action, dimension of each gate's weight matrix is (num_experts, input_size)
        self.gates = nn.ModuleList([nn.Linear(input_size, self.num_experts) for _ in range(self.action_size)])
        # experts, dimension of each expert's weight matrix is (expert_output_dim, input_size)
        self.experts = nn.ModuleList([nn.Linear(input_size, self.expert_output_dim) for _ in range(self.num_experts)])
        # last layers, one for each action
        self.last_layers = nn.ModuleList(
            [nn.Linear(self.expert_output_dim, 1) for _ in range(self.action_size)]
        )

        # Initializing the neural-network
        # initializing experts
        for i in range(self.num_experts):
            nn.init.kaiming_normal_(self.experts[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.experts[i].bias)
        # initializing last layers
        for k in range(self.action_size):
            nn.init.normal_(self.last_layers[k].weight, mean=0., std=.0001)
            nn.init.zeros_(self.last_layers[k].bias)
        return

    def forward(self, state):
        """
        :purpose: implement forward pass.
        :param state: {Tensor(batch_size, 9)}
        :return: action_params {Tensor(batch_size, 3)}
        """
        negative_slope = 0.01
        x = state

        # first pass x through each gate and each expert.
        gate_outputs = [F.softmax(self.gates[k](x), dim=-1) for k in range(self.action_size)]
        if self.activation == "relu":
            expert_outputs = [F.relu(self.experts[i](x)) for i in range(self.num_experts)]
        elif self.activation == "leaky_relu":
            expert_outputs = [F.leaky_relu(self.experts[i](x), negative_slope) for i in range(self.num_experts)]
        else:
            raise ValueError("Unknown activation function " + str(self.activation))

        # compute the output for each action k by using gate_outputs[k] and expert_outputs
        mmoe_outputs = []
        for k in range(self.action_size):
            out = 0
            gk = gate_outputs[k]  # {Tensor(batch_size, self.num_experts)} or {Tensor(self.num_experts)}
            for i in range(self.num_experts):
                gk_i = gk[:, i][:, None]  # {Tensor: shape=(batch_size, 1)}
                out += gk_i * expert_outputs[i]
            mmoe_outputs.append(out)

        # last layer outputs (with ReLU activation)
        last_layer_outputs = [self.last_layers[k](mmoe_outputs[k]) for k in range(self.action_size)]

        # concatenate the outputs (one parameter for each action)
        action_params = torch.cat(tuple(last_layer_outputs), dim=1)
        # action_params = torch.tanh(action_params)  # restricting values within range taken care of by inverting
        # gradients and small weight initialization of last layers.
        return action_params


# %-----------------------------------------------------------------------------------%
# %--------------------- CLASS IMPLEMENTATION OF PADQNAgent --------------------------%
# %-----------------------------------------------------------------------------------%
class PADQNAgent(Agent):
    NAME = "PA-DQN Agent"

    def __init__(
            self,
            observation_space,
            action_space,
            seed=1,
            batch_size=128,
            gamma=0.9,
            replay_memory_size=10000,
            initial_memory_threshold=0,
            epsilon_initial=1.0,
            epsilon_steps=1000,
            epsilon_final=0.01,
            actor_class=QActor,
            actor_param_class=ParamActor,
            alpha_actor=0.001,
            alpha_actor_param=0.0001,
            tau_actor=0.1,  # polyak averaging factor for copying target weights
            tau_actor_param=0.001,
            actor_kwargs={},
            actor_param_kwargs={},
            loss='l1_smooth',  # F.mse_loss or F.smooth_l1_loss
            clip_grad=10.,
            use_ornstein_noise=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(PADQNAgent, self).__init__(observation_space, action_space)
        self.name = 'PADQN'
        # -------- set seed for deterministic behavior -------
        self.np_random = None
        self.seed = seed
        self._seed(self.seed)
        self.device = torch.device(device)

        # ------ copy attributes
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.epsilon = epsilon_initial
        self.alpha_actor = alpha_actor
        self.alpha_actor_param = alpha_actor_param
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self.loss = loss
        self.use_ornstein_noise = use_ornstein_noise
        self.clip_grad = clip_grad

        # ------ set important attributes
        self._step = 0  # track steps inside an episode
        self._episode = 0  # track epsilon for annealing it over episodes
        self.updates = 0  # number of times agent optimizes over td based loss
        # number of actions
        self.num_actions = self.action_space.spaces[0].n  # number of actions, equals 3 for platform
        # list of parameter shapes, one shape for each action. each shape equals 1 for platform
        self.action_parameter_sizes = np.array(
            [self.action_space.spaces[i].shape[0] for i in range(1, self.num_actions + 1)]
        )  # [1, 1, 1]
        # sum of all parameter dimensions, equals 3 for platform
        self.action_parameter_size = int(self.action_parameter_sizes.sum())  # 3
        # maximum value, minimum value, and ranges of parameters
        self.action_parameter_max_numpy = np.concatenate(
            [self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate(
            [self.action_space.spaces[i].low for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)

        # ------ Ornstein-Uhlenback Noise
        self.noise = OrnsteinUhlenbeckActionNoise(
            self.action_parameter_size,
            mu=0.,
            theta=0.15,
            sigma=0.0001,
            random_machine=self.np_random
        )

        # --------- instantiate replay memory, q_actor, and param_actor
        # # ------ instantiate replay memory
        self.replay_memory = Memory(
            replay_memory_size,
            observation_space.shape,  # 9
            (1 + self.action_parameter_size,),  # 4
            next_actions=False
        )
        # # -------- instantiate q_actor
        self.actor = actor_class(
            self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs
        ).to(device)
        # # -------- actor's target
        self.actor_target = actor_class(
            self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs
        ).to(device)
        # copy parameters to target
        hard_update_target_network(self.actor, self.actor_target)
        # keep target in eval mode
        self.actor_target.eval()
        # configure loss function (for q_actor)
        if self.loss == 'l1_smooth':
            self.loss_func = F.smooth_l1_loss
        elif self.loss == 'mse':
            self.loss_func = F.mse_loss
        else:
            raise ValueError("Unknown loss function " + self.loss)
        # configure optimizer
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.alpha_actor)
        # # ------- instantiate param_actor
        self.actor_param = actor_param_class(
            self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_param_kwargs
        ).to(device)
        # actor_param's target
        self.actor_param_target = actor_param_class(
            self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_param_kwargs
        ).to(device)
        # copy parameters to target
        hard_update_target_network(self.actor_param, self.actor_param_target)
        # keep target in eval mode
        self.actor_param_target.eval()
        # configure optimizer
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.alpha_actor_param)

    def __str__(self):
        """
        :purpose: print description of the agent.
        :return:
        """
        desc = super().__str__() + "\n"
        desc += "QActor Network: {}\n".format(self.actor) + \
                "ParamActor Network: {}\n".format(self.actor_param) + \
                "Seed: {}\n".format(self.seed) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Replay Memory Size: {}\n".format(self.replay_memory_size) + \
                "Initial Memory Threshold: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Alpha (QActor): {}\n".format(self.alpha_actor) + \
                "Alpha (ParamActor): {}\n".format(self.alpha_actor_param) + \
                "Tau (QActor): {}\n".format(self.tau_actor) + \
                "Tau (ParamActor): {}\n".format(self.tau_actor_param) + \
                "Loss: {}\n".format(self.loss) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Clip Grad: {}".format(self.clip_grad)
        return desc

    def _seed(self, seed=1):
        """
        :purpose: Set seed for deterministic behavior.
        :param seed:
        :return:
        """
        np.random.seed(seed)  # numpy random module.
        self.np_random = np.random.RandomState(seed=seed)
        random.seed(seed)  # python random module.
        torch.manual_seed(seed)  # torch seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return

    def start_episode(self):
        """
        :purpose: overload start_episode of Agent class.
        """
        pass

    def end_episode(self):
        """
        :purpose: Overload end_episode function of Agent class. Importantly, update epsilon for next episode.
        """
        self._episode += 1

        ep = self._episode
        if ep < self.epsilon_steps:
            # linear annealing of epsilon
            self.epsilon = self.epsilon_initial + (
                    (self.epsilon_final - self.epsilon_initial) / self.epsilon_steps) * ep
        else:
            self.epsilon = self.epsilon_final
        return

    def act(self, state):
        """
        :purpose: overload act function of Agent class.
        :param state: {ndarray(9,)}
        :return: action, action_parameters, all_action_parameters
         - action {int}: lies in {0, 1, 2}.
         - action_parameters {ndarray(1,)}:
         - all_action_parameters {ndarray(3,)}:
        """
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)  # Tensor(9,)
            all_action_parameters = self.actor_param.forward(state.unsqueeze(0))  # Tensor(1, 3)

            # epsilon-greedy discrete actions
            rnd = self.np_random.uniform()
            if rnd < self.epsilon:
                # random action
                action = self.np_random.choice(self.num_actions)
            else:
                # greedy action (select action with maximum Q-value)
                # unsqueeze state and all_action_parameters before passing through actor
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters)
                Q_a = Q_a.detach().cpu().numpy()
                action = np.argmax(Q_a)

            # convert all_action_parameters to ndarray(3,)
            all_action_parameters = all_action_parameters.cpu().numpy().squeeze(0)  # ndarray(3,)
            # add noise only to parameter of chosen action (for exploration in continuous space)
            if self.noise is not None:
                sigma = 0.001
                all_action_parameters[action] += self.noise.sample()[action] if self.use_ornstein_noise else \
                    sigma * self.np_random.randn(self.num_actions)[action]
            # chosen action's parameter
            action_parameters = np.array([all_action_parameters[action]])

        return action, action_parameters, all_action_parameters

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        """
        :purpose: step over to the next time-step and learn.
        :param state: {ndarray(9,)}
        :param action: {tuple(2)}
            - action[0]: {int}
            - action[1]: {ndarray(3,)}
        :param reward: {float64}
        :param next_state: {ndarray(9,)}
        :param next_action: {tuple(2)}
            - next_action[0]: {int}
            - next_action[1]: {ndarray(3,)}
        :param terminal: {bool}
        :param time_steps: {int}
            - not used.
        :return:
        """
        act, all_action_parameters = action  # act {int}, all_action_parameters {ndarray(3,)}
        self._step += 1

        # add the sample to agent's replay memory
        self._add_sample(
            state,
            np.concatenate(([act], all_action_parameters)),  # storing as ndarray(4,), see _add_sample's docstring.
            reward,
            next_state,
            np.concatenate(([next_action[0]], next_action[1])),  # storing as ndarray(4,)
            terminal
        )
        if self._step >= max(self.batch_size, self.initial_memory_threshold):
            self._optimize_td_loss()
            self.updates += 1

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        """
        :purpose: add the given sample to agent's replay memory. In memory, we store the action as ndarray(4,)
        object where first position corresponds to the discrete action and the next three are the action parameters for
        all actions.
        :param state: {ndarray(9,)}
        :param action: {ndarray(4,)}
        :param reward: float64
        :param next_state: {ndarray(9,)}
        :param next_action: {ndarray(4,)}
        :param terminal: bool
        :return:
        """
        assert len(action) == 1 + self.action_parameter_size
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)
        return

    def _optimize_td_loss(self):
        """
        :purpose: Update Q network and Action-Parameters Network parameters
        :return:
        """
        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(
            self.batch_size,
            random_machine=self.np_random
        )
        # convert to tensors and get the desired dimensions
        states = torch.from_numpy(states).to(self.device)  # {Tensor(batch_size, 9)}
        actions_combined = torch.from_numpy(actions).to(self.device)  # Tensor(batch_size, 4)
        # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()  # {Tensor(batch_size), dtype=int64}
        action_parameters = actions_combined[:, 1:]  # {Tensor(batch_size, 3)}
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()  # Tensor(batch_size,)
        next_states = torch.from_numpy(next_states).to(self.device)  # Tensor(batch_size, 9)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()  # Tensor(batch_size,)

        # ---------------------- optimize Q-network ----------------------
        # compute target for QActor (Q values obtained from QActor's target using action parameters from ParamActor's
        # target)
        with torch.no_grad():
            # get next action parameters from ParamActor's target
            pred_next_action_parameters = self.actor_param_target.forward(next_states)  # {Tensor(batch_size, 3)}
            # get Q values from QActor's target
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)  # {Tensor(batch_size, 3)}
            # compute the maximum over Q values
            Q_prime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()  # {Tensor(batch_size)}
            # compute the target {Tensor(batch_size,)}
            # target = reward + (1 - terminal) * gamma * max {Q(next_state, a, theta_x(a); theta_Q-): a = 0, 1, 2}
            target = rewards + (1 - terminals) * self.gamma * Q_prime  # {Tensor(batch_size)}

        # compute prediction (QActor's Q values) {Tensor(batch_size,)}
        q_values = self.actor(states, action_parameters)  # {Tensor(batch_size, 3)}
        y_predicted = q_values.gather(1, actions.unsqueeze(-1)).squeeze()  # {Tensor(batch_size,)}
        loss_Q = self.loss_func(y_predicted, target)

        # update QActor
        self.actor_optimiser.zero_grad()  # zero the gradients
        loss_Q.backward()  # compute the gradients
        if self.clip_grad > 0:  # clip the gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()  # update

        # ---------------------- optimize actor ----------------------
        # first we compute gradients of mean_return with
        # respect to action-parameters (output of ParamActor, i.e., x(.;\theta_x)) we will then alter these gradients
        # using inverting gradients approach [see https://arxiv.org/abs/1511.04143, page 7] the altered gradients
        # are the ones that should be used to update the weights of ParamActor (\theta_x)
        with torch.no_grad():
            # compute action_params using ParamActor
            action_params = self.actor_param(states)  # {Tensor(batch_size, 3)}
        # enable gradients for tracking mean_return
        action_params.requires_grad = True
        # compute Q values from QActor
        Q = self.actor(states, action_params)  # {Tensor(batch_size, 3)}
        # compute mean_return
        mean_return = torch.mean(torch.sum(Q, 1))  # {Tensor()}

        # compute gradients of mean_return with respect to action_params
        self.actor.zero_grad()
        mean_return.backward()
        # create a copy of computed gradients
        from copy import deepcopy
        delta_x = deepcopy(action_params.grad.data)  # {Tensor(batch_size, 3)}
        # now alter the gradients
        action_params = self.actor_param(states)
        delta_x = self._invert_gradients(delta_x, action_params)  # {Tensor(batch_size, 3)}
        # define dummy function out = delta_x * action_params
        # (its gradient over action_params is simply delta_x)
        out = -torch.mul(delta_x, action_params)  # {Tensor(batch_size, 3)}

        # update ParamActor
        self.actor_param.zero_grad()  # zero the gradients
        out.backward(torch.ones(out.shape).to(self.device))  # compute gradients
        if self.clip_grad > 0:  # clip the gradients
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)
        self.actor_param_optimiser.step()  # update

        # ---------------------- update targets ----------------------
        # soft update of target networks of QActor and ParamActor
        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)
        return

    def _invert_gradients(self, grad, vals):
        """
        :purpose:
        :param grad:
        :param vals:
        :return:
        """
        max_p = self.action_parameter_max
        min_p = self.action_parameter_min
        rnge = self.action_parameter_range

        grad = grad.to(self.device)
        vals = vals.to(self.device)

        assert grad.shape == vals.shape

        with torch.no_grad():
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), prefix + '_actor.pt')
        torch.save(self.actor_param.state_dict(), prefix + '_actor_param.pt')
        print('Models saved successfully.')

    def load_models(self, prefix):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param prefix: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(
            torch.load(prefix + '_actor.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.actor_param.load_state_dict(
            torch.load(prefix + '_actor_param.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        print('Models loaded successfully.')
