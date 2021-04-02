import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class IAMPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(IAMPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = IAMImageBase
            elif len(obs_shape) == 1:
                if base_kwargs['env_name'] == 'warehouse':
                    base = IAMWarehouseBase
                    if base_kwargs['IAM']:
                        hxs_size = 25
                    else:
                        hxs_size = 73
                elif base_kwargs['env_name'] == 'traffic':
                    base = IAMTrafficBase
                    if base_kwargs['IAM']:
                        hxs_size = 4
                    else:
                        hxs_size = 30
            else:
                raise NotImplementedError

        # self.base = base(obs_shape[0], **base_kwargs)
        self.base = base(obs_shape[0], hxs_size)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

class IAMBase(nn.Module):
    """
    Influence-Aware Memory archtecture

    NOTE: Implement later as a base for different tasks
    """
    def __init__(self, num_predictors, hidden_size_gru):
        super(IAMBase, self).__init__()

        self.gru = nn.GRU(num_predictors, hidden_size_gru)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        
    @property
    def is_recurrent(self):
        return True

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class IAMWarehouseBase(IAMBase):
    """
    IAM architecture for Warehouse environment

    obs ->  |fnn |            ->|-> |nn  | ->critic_linear()->value
            |____|              |   |____|
                                |   
        ->  |dset|  -> |gru | ->|-> |nn  | ->dist()->mode()/sample()->action 
            |____|     |____|       |____|
    
    NOTE:
    observation: (num_processes, num_inputs: 73 in warehouse)
    fnn output: (num_processes, hidden_size_fnn)
    dset output: (num_processes, 25)
    gru output: ((num_processes, hidden_size_gru), rnn_hxs)
    output_size:  hidden_size_fnn plus hidden_size_gru
    """
    def __init__(self, num_inputs, hidden_size=64):
        super(IAMWarehouseBase, self).__init__(25, hidden_size)

        self._hidden_size_gru = hidden_size
        self._hidden_size_fnn = hidden_size
        # NOTE the prior knowledge: 
        # the hidden state does not depends on the position, thus manually 
        # extracting the subset of the observation
        self._dset = np.array([0, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]).astype(int)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.fnn = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        # self.critic_linear = \
        #             init_(nn.Linear(self._hidden_size_fnn + self._hidden_size_gru, 1))
        self.actor = nn.Sequential(
            init_(nn.Linear(2*hidden_size, 2*hidden_size)),nn.Tanh(),
            init_(nn.Linear(2*hidden_size, 2*hidden_size)),nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(2*hidden_size, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        self.train()

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size_gru

    @property
    def output_size(self):
        return self._hidden_size_fnn + self._hidden_size_gru

    def forward(self, inputs, rnn_hxs, masks):
        x = self.fnn(inputs)
        
        d_obs = self._dset_extraction(inputs)
        d, rnn_hxs = self._forward_gru(d_obs, rnn_hxs, masks)

        x = torch.cat((x, d), dim=1)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return hidden_critic, hidden_actor, rnn_hxs

    def _dset_extraction(self, obs):
        d_obs = obs[:, self._dset]
        return d_obs

class IAMTrafficBase(IAMBase):
    """
    IAM architecture for traffic control environment

    obs ->  |fnn |            ->|-> |nn  | ->critic_linear()->value
            |____|              |   |____|
                                |   
        ->  |dset|  -> |gru | ->|-> |nn  | ->dist()->mode()/sample()->action 
            |____|     |____|       |____|

    NOTE:
        dset output: (num_processes, 4)
    """
    def __init__(self, num_inputs, hidden_size=64):
        super(IAMTrafficBase, self).__init__(4, hidden_size)
        self._hidden_size_gru = hidden_size
        self._hidden_size_fnn = hidden_size

        # NOTE the prior knowledge: 
        self._dset = np.array([13, 14, 28, 29]).astype(int)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.fnn = nn.Sequential(
            init_(nn.Linear(num_inputs, 4*hidden_size)),nn.ReLU(),
            init_(nn.Linear(4*hidden_size, hidden_size)),nn.ReLU())

        # self.critic_linear = \
        #             init_(nn.Linear(self._hidden_size_fnn + self._hidden_size_gru, 1))
        self.actor = nn.Sequential(
            init_(nn.Linear(2*hidden_size, 2*hidden_size)),nn.Tanh(),
            init_(nn.Linear(2*hidden_size, 2*hidden_size)),nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(2*hidden_size, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        self.train()

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size_gru

    @property
    def output_size(self):
        return self._hidden_size_fnn + self._hidden_size_gru

    def forward(self, inputs, rnn_hxs, masks):
        x = self.fnn(inputs)
        
        d_obs = self._dset_extraction(inputs)
        d, rnn_hxs = self._forward_gru(d_obs, rnn_hxs, masks)

        x = torch.cat((x, d), dim=1)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return hidden_critic, hidden_actor, rnn_hxs

    def _dset_extraction(self, obs):
        d_obs = obs[:, self._dset]
        return d_obs

class IAMImageBase(IAMBase):
    """
    IAM architecture for image observed environment

    obs -> |cnn | -> |-> flatten() -> |fnn |   ->|-> |nn  | ->critic_linear()->value
           |____|    |                |____|     |   |____|
                     |    |atte|                 |
                     |->  |tion|   -> |gru |   ->|-> |nn  | ->dist()->mode()/sample()->action 
                          |____|      |____|         |____|   
    """
    def __init__(self, num_inputs, hidden_size=128):
        super(IAMImageBase, self).__init__(113, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
                               
        self._hidden_size_gru = hidden_size
        self._hidden_size_fnn = hidden_size

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU())

        self.fnn = nn.Sequential(
            Flatten(),
            init_(nn.Linear(64 * 7 * 7, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic_linear = \
                    init_(nn.Linear(self._hidden_size_fnn + self._hidden_size_gru, 1))

        # self.actor = nn.Sequential(
        #     init_(nn.Linear(2*hidden_size, 2*hidden_size)),nn.Tanh(),
        #     init_(nn.Linear(2*hidden_size, 2*hidden_size)),nn.Tanh())

        # self.critic = nn.Sequential(
        #     init_(nn.Linear(2*hidden_size, hidden_size)),nn.Tanh(),
        #     init_(nn.Linear(hidden_size, hidden_size)),nn.Tanh(),
        #     init_(nn.Linear(hidden_size, 1)))

        # Layers for attention
        self.dpatch_conv = init_(nn.Linear(64, 128)) #depatch, merge the channels and encode them

        # self.dpatch_auto = init_(nn.Linear(64, 128))
        # self.dpatch_auto_norm = init_(nn.Linear(7*7*128, 128))

        self.dpatch_prehidden = init_(nn.Linear(self._hidden_size_gru, 128))

        self.dpatch_combine = nn.Tanh()

        self.dpatch_weights = nn.Sequential(
            init_(nn.Linear(128,1)), nn.Softmax(dim=1))

        self.train()

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size_gru

    @property
    def output_size(self):
        return self._hidden_size_fnn + self._hidden_size_gru

    def forward(self, inputs, rnn_hxs, masks):
        hidden_conv = self.cnn(inputs / 255.0)

        x = self.fnn(hidden_conv)

        inf_hidden = self.attention(hidden_conv, rnn_hxs)
        d, rnn_hxs = self._forward_gru(inf_hidden, rnn_hxs, masks)

        x = torch.cat((x, d), dim=1)

        # hidden_critic = self.critic(x)
        # hidden_actor = self.actor(x)

        # return hidden_critic, hidden_actor, rnn_hxs
        return self.critic_linear(x), x, rnn_hxs

    def attention(self, hidden_conv, rnn_hxs):
        hidden_conv = hidden_conv.permute(0,2,3,1)
        shape = hidden_conv.size()
        num_regions = shape[1]*shape[2]
        hidden = torch.reshape(hidden_conv, ([-1,num_regions,shape[3]]))
        linear_conv = self.dpatch_conv(hidden)        
        linear_prehidden = self.dpatch_prehidden(rnn_hxs)
        # print(linear_prehidden.size())
        context = self.dpatch_combine(linear_conv + torch.unsqueeze(linear_prehidden, 1))
        attention_weights = self.dpatch_weights(context)
        dpatch = torch.sum(attention_weights*hidden,dim=1)
        inf_hidden = torch.cat((dpatch,torch.reshape(attention_weights, ([-1, num_regions]))), 1)

        return inf_hidden