from pettingzoo.mpe import simple_spread_v3
from src.MADDPG import MADDPG
import torch

if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    M = 4096
    length = 50
    env = simple_spread_v3.parallel_env(render_mode="human",max_cycles=length)
    control = MADDPG(54, 5, 3, 0.95, 1.1, 0.1, 32, device)


    for episode in range(M):
        observations, infos = env.reset()
        obs = []
        for k in observations:
            obs.extend(observations[k])
        obs = torch.tensor(obs).to(device)
        obs_new = obs

        for t in range(length):
            obs = obs_new
            act = control.get_action(obs)
            actions = {agent: action for agent,action in zip(env.agents,act)}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            r = []
            with torch.no_grad():
                for k in rewards:
                    r.append(rewards[k])
                r = torch.tensor(r).to(device)
            obs_new = obs

            control.add_to_replay(observations, torch.tensor(act), torch.tensor(r), observations)
            control.update()
    env.close()