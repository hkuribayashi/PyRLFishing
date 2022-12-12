import gym

from stable_baselines3 import DQN

env = gym.make('gym_phishing:RLPTraining-v0')
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_cartpole")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

env = gym.make('gym_phishing:RLPTest-v0')
model.set_env(env)
obs = env.reset()
contador = 0
media = 0.0
for _ in range(3000):
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if info['acertou']:
            contador += 1
        if done:
            obs = env.reset()
    media += contador / 1000
    print("Acurárica: {}".format(contador / 1000))

print(media/3000)
