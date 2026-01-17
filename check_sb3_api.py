import stable_baselines3
from stable_baselines3 import PPO
import inspect

print(f"SB3 Version: {stable_baselines3.__version__}")
print("\nPPO.__init__ signature:")
sig = inspect.signature(PPO.__init__)
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
