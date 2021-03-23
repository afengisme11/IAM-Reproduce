# IAM-Reproduce
This repo is for the Influence-Aware Memory(IAM) architecture

## Run

Currently achieved the scenario of warehouse, to run, use
```bash
python main.py --env-name Warehouse --num-processes 16
```
To visualize, alter the variable render_bool to True  in warehouse.py, and run with just 1 processes(recommended, all processes will pop out):
```
python main.py --env-name Warehouse --num-processes 1
```