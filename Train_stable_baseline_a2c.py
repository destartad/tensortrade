#%% Set up env
"""
from stable_baselines3 import A2C

from MT_env import create_env
env = create_env()

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="log",)
model.learn(total_timesteps=400000)
model.save("a2c_exp")


"""#%%
from stable_baselines3 import A2C
from MT_env import create_env
env = create_env()
model = A2C.load("a2c_exp")
dones = False
obs = env.reset()
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# %%
