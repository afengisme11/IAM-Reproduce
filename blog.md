# Influence-aware Memory Architecture


## Reproduce project of the IAM paper


# Authors

*   Jianfeng Cui
*   Zheyu Du

# Remark

This blog has been submitted to https://reproducedpapers.org, which features a collection of reproducibility attempts of papers in the field of Deep Learning by various people. If you are interested, feel free to check it out!

# Introduction

![Screenshot from 2021-04-02 20-22-07](blog.assets/Screenshot from 2021-04-02 20-22-07.png)

# Network Architecture

![Screenshot from 2021-04-02 20-18-13](blog.assets/Screenshot from 2021-04-02 20-18-13.png)

## Warehouse

## Traffic Control


## Working Scenario - Flickering Atari

In a more complex working scenario, the input features that we want our algorithm to focus on may change rapidly over time. For example, in a 'Breakout' video game, what we want our algorithm to know is the location of the ball. However, it's changing rapidly and we cannot determine where the ball is for a specific time step without the knowledge of previous states. In such cases, attention mechanism should be used instead of manually selected d-set. Thus, we implemented it in the IAM structure to deal with such problems. 

A new class is defined for this task which also inherits the recurrent function from class 'IAMBase'.  Additionally, a convolutional neural network is defined for image processing and a fully connected neural network is defined for the IAM. The entire structure of the neural network graph used in this case is also shown.

```python
class atariBase(IAMBase):
    """
    IAM architecture for image observed environment

    obs -> |cnn | -> |-> flatten() -> |fnn |   ->|-> |nn  | ->critic_linear()->value
           |____|    |                |____|     |   |____|
                     |    |atte|                 |
                     |->  |tion|   -> |gru |   ->|-> |nn  | ->dist()->mode()/sample()->action 
                          |____|      |____|         |____|   
    """
    def __init__(self, num_inputs, hxs_size, recurrent=False, IAM=False, hidden_size=64):
        super(atariBase, self).__init__(recurrent, IAM, hxs_size, hidden_size)
        self._depatch_size = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU())

        self.fnn = nn.Sequential(
            Flatten(),
            init_(nn.Linear(64 * 7 * 7, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.ReLU())

```

Two linear layers are defined for the outputs and some functional layers here will be used in the attention mechanism which we will introduce later.

```python
init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
                               
        self.actor = nn.Sequential(
            init_(nn.Linear(2*hidden_size, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(2*hidden_size, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        # functional layers
        self.dpatch_conv = init_(nn.Linear(64, 128)) #depatch, merge the channels and encode them

        self.dpatch_auto = init_(nn.Linear(64, 128))
        self.dpatch_auto_norm = init_(nn.Linear(7*7*128, 128))

        self.dpatch_prehidden = init_(nn.Linear(hidden_size, 128))

        self.dpatch_combine = nn.Tanh()

        self.dpatch_weights = nn.Sequential(
            init_(nn.Linear(128,1)), nn.Softmax(dim=1))

        self.train()
```
Now we define the function for the attention mechanism, which is the most important part of our controller. This function takes two input. The 'hidden_conv' is the output of the convolution neural network and the 'rnn_hxs' is the hidden state of the recurrent neural network from the last time step. The new observation tensor will first be reshaped into [batch size, width, height, channels] and the dimensions of width and height will be merged into one which represents the regions on every channels of the input.  Next, the weights matrix of all the regions will be calculated from the combination of current input and past hidden state. Finally, the weight matrix is used to decide which regions of the input observations should be passed into the recurrent neural network at current time step. Intuitively speaking, our algorithm will learn where to look at according to the memory of the past states.
```python
def attention(self, hidden_conv, rnn_hxs):
        hidden_conv = hidden_conv.permute(0,2,3,1)
        shape = hidden_conv.size()
        num_regions = shape[1]*shape[2]
        hidden = torch.reshape(hidden_conv, ([-1,num_regions,shape[3]]))
        linear_conv = self.dpatch_conv(hidden)        
        linear_prehidden = self.dpatch_prehidden(rnn_hxs)
        context = self.dpatch_combine(linear_conv + torch.unsqueeze(linear_prehidden, 1))
        attention_weights = self.dpatch_weights(context)
        dpatch = torch.sum(attention_weights*hidden,dim=1)
        inf_hidden = torch.cat((dpatch,torch.reshape(attention_weights, ([-1, num_regions]))), 1)

        return inf_hidden
```
In the code below, we show how our controller reacts to the new observations. When new inputs come in (which are images), they will first go through a CNN. Next, the output will go through both a FNN after being flattened and a RNN after being filtered by the attention function. Then the outputs are concatenated and go through two separate FNN. Finally, we get the critic and actor for the RL algorithm. The definition of the neural networks used here could be found above.
```python
def forward(self, inputs, rnn_hxs, masks):
        hidden_conv = self.cnn(inputs / 255.0)

        fnn_out = self.fnn(hidden_conv)
        inf_hidden = self.attention(hidden_conv, rnn_hxs)
        rnn_out, rnn_hxs = self._forward_gru(inf_hidden, rnn_hxs, masks)

        x = torch.cat((rnn_out,fnn_out), 1)

        hidden_critic = self.critic(x) 
        hidden_actor = self.actor(x)

        return hidden_critic, hidden_actor, rnn_hxs
```















# Experiment (plots and analysis)


# Summary (accomplishment, drawbacks, improvement)
