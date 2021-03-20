from stable_baselines3.common.env_checker import check_env
from environments.warehouse.warehouse import Warehouse
import yaml

with open("default_warehouse.yaml", 'r') as stream:
    try:
        parameters = yaml.safe_load(stream)['parameters']
    except yaml.YAMLError as exc:
        print(exc)

env = Warehouse(seed=1, parameters=parameters)
# It will check your custom environment and output additional warnings if needed
check_env(env)