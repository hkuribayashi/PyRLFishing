from gym.envs.registration import register

register(
    id="RLPTraining-v0",
    entry_point="gym_phishing.envs:RLPTraining"
)

register(
    id="RLPTest-v0",
    entry_point="gym_phishing.envs:RLPTest"
)