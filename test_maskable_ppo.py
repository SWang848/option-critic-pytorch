from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks


from crafting import CraftingEnv, MineCraftingEnv, RandomCraftingEnv
from crafting.task import (
    RewardShaping,
    TaskObtainItem,
    adaptative_max_episode_step,
    get_task,
)


# env = MineCraftingEnv(max_step=int(1e6), seed=1)
# task = TaskObtainItem(env.world,env.world.item_from_name["wood_plank"],reward_shaping=RewardShaping(RewardShaping.DIRECT_USEFUL))
# env.add_task(task)
# action_masks = get_action_masks(env)
# print(action_masks)

env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
model.learn(5000)

evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=80, warn=False)

model.save("ppo_mask")
del model # remove to demonstrate saving and loading

model = MaskablePPO.load("ppo_mask")

obs = env.reset()
while True:
    # Retrieve current action mask
    action_masks = get_action_masks(env)
    print(action_masks)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, rewards, dones, info = env.step(action)
    env.render()