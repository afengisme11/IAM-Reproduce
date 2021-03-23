# IAM-Reproduce
This repo is for the Influence-Aware Memory(IAM) architecture(https://arxiv.org/abs/1911.07643), based on the pytorch structure of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and the paper's repo source https://github.com/INFLUENCEorg/influence-aware-memory

## Run

Currently only achieved the scenario of warehouse, to run, use
```bash
cd YOUR_PATH/IAM-Reproduce
python main.py --env-name Warehouse --num-processes 16 --num-env-steps 4000000 --num-steps 10 --log-dir ./log
```
The `log`folder will store the monitor files of all processes and a manually stored file `mean_rewards.txt` recording the mean rewards. To visualize the results, use

```bash
python plot_results.py
```

Currently the result of mean rewards is like:

![warehouse](README.assets/warehouse.png)

To render the warehouse dynamics, alter the variable `render_bool` to True  in `warehouse.py`, and run with just 1 processes(recommended, because all processes will pop out):

```bash
python main.py --env-name Warehouse --num-processes 1
```

## Work

Currently the work is customized in:

1. Alter the `warehouse.py` and somewhere else to embed it in the pytorch structure
2. Design `IAMModel.py` for Influence-Aware Model, used with A2C now

3. Visualize the result