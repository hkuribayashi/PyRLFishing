import os
from config.default import RLConfig
from rl.A2C import A2C
from rl.DQN import DQN
from rl.PPO import PPO

os.environ['KMP_DUPLICATE_LIB_OK']='True'
config = RLConfig(600, 0.0001, [64, 64], 100000, 0)

simulacao1 = DQN(1)
simulacao1.train()
simulacao1.test()

simulacao2 = A2C(2)
simulacao2.train()
simulacao2.test()

simulacao3 = PPO(3)
simulacao3.train()
simulacao3.test()
