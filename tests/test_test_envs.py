import gym
import test_envs

if __name__ == "__main__":
    env_ids = [
        "MovementBandits-v0",
        "KeyDoor-v0",
        "Allwalk-v0",
        "Fourrooms-v0",
    ]

    for env_id in env_ids:
        print(env_id)
        env = gym.make(env_id)
        env.reset()
        total_reward = 0
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())
            total_reward += reward
        print("Total Reward:", total_reward)
        env.close()
