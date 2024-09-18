import numpy as np
import tensorflow as tf
import os
import contextlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import utils.Utils as Utils
import pickle
from tensorflow.keras.callbacks import LambdaCallback

# 创建LambdaCallback实例
log_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: None)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

loaded_model = tf.keras.models.load_model('saved_model/custom_dqn_model12.h5')
loaded_model2 = tf.keras.models.load_model('saved_model/dqn_model_for_win9.h5')
 
# 加载代理的状态
with open('saved_model/custom_dqn_agent12.pkl', 'rb') as f:
    loaded_agent_dict = pickle.load(f)
with open('saved_model/dqn_model_agent_for_win9.pkl', 'rb') as f:
    loaded_agent_dict2 = pickle.load(f)
# 定义神经网络模型
def build_model(input_shape, output_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(output_shape, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model

# 创建环境
class MahjongEnvironment:
    def __init__(self):
        self.state = np.zeros(12, dtype=int)
        self.opponentState = np.zeros(12, dtype=int)
        self.done = False
        self.reward = 0
        self.info = {}
        self.deck = np.repeat(np.arange(0, 12), 4).tolist()
        self.index = 0
        np.random.shuffle(self.deck)

    def put(self, action):
        self.state[action] += 1

    def putOppo(self, action):
        self.opponentState[action] += 1

    def getAvailAction(self, action):
        a = -100
        for i in range(12):
            if (self.state[i] > 0 and np.abs(action - i) < np.abs(a - i)):
                a = i
        return a
    
    def getOppoAvailAction(self, action):
        a = -100
        for i in range(12):
            if (self.opponentState[i] > 0 and np.abs(action - i) < np.abs(a - i)):
                a = i
        return a

    def reset(self):
        for i in range(4):
            self.put(self.deck[self.index])
            self.index += 1
            self.putOppo(self.deck[self.index])
            self.index += 1
        self.done = False
        return self.state, self.opponentState

    def step(self, agent, oppo_agent):
        if(self.index >= 48):
            self.reward = 0
            self.done = True
            return cur_state, self.state, action, self.reward, self.done, self.info
        self.put(self.deck[self.index])
        cur_state = np.copy(self.state)
        # with open('training_log.txt', 'a') as f:
        #     with contextlib.redirect_stdout(f):
        #         print(Utils.showHand(self.state))
        self.index += 1
        self.reward = Utils.is_hand_finished(self.state)
        # if (self.reward > 0):
        #     self.reward = 1
        # else:
        #     self.reward = -1
        self.done = self.reward > 0
        action = 0
        if (self.done == False):
            action = agent.act(self.state)
            # with open('training_log.txt', 'a') as f:
            #     with contextlib.redirect_stdout(f):
            # # 执行动作并返回新的状态、奖励和是否完成
            #         print("action: " + action.__str__())
            if(self.index >= 48):
                self.reward = 0
                self.done = True
                return cur_state, self.state, action, self.reward, self.done, self.info
            self.state[self.getAvailAction(action)] -= 1  # 出牌
            self.putOppo(self.deck[self.index])
            opporeward = Utils.is_hand_finished(self.opponentState)
            if (opporeward > 0):
                self.reward = -opporeward
                self.done  = True
                return cur_state, self.state, action, self.reward, self.done, self.info
            action2 = oppo_agent.act(self.opponentState)
            # action2 = np.random.randint(12)
            self.opponentState[self.getOppoAvailAction(action2)] -= 1  # 出牌
        return cur_state, self.state, action, self.reward, self.done, self.info

# 创建DQN代理
class DQNAgent:
    def __init__(self, state_size, action_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.model = None #build_model(state_size, action_size)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.timesteps_since_update = 0
        self.batch_size = batch_size
        self.update_frequency = 256
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def remember(self, state, action, reward, next_state, done):
        # 存储经验
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 根据当前状态选择动作
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state[np.newaxis, :],verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < self.batch_size:
            return
        # 从记忆中采样并训练模型
        minibatch = np.random.choice(len(self.memory), size=batch_size, replace=True)
        for state, action, reward, next_state, done in [self.memory[i] for i in minibatch]:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state[np.newaxis, :],verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)
            target_f[0][action] = target
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0, callbacks=[log_callback])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 主训练循环
def train_agent(agent, oppo_agent, episodes):
    wins = 0
    totalrew = 0
    for e in range(episodes):
        env = MahjongEnvironment()
        state, state2 = env.reset()
        done = False
        list = []
        while not done:
            cur_state, state, action, reward, done, _ = env.step(agent, oppo_agent)
            # with open('training_log.txt', 'a') as f:
            #     with contextlib.redirect_stdout(f):
            #         print("hand: " + Utils.showHand(state).__str__())
            #         print("reward： " + reward.__str__())
            # agent.remember(cur_state, action, reward, state, done)
            list.append([np.copy(cur_state), action, reward, np.copy(state), done])
            # print(list)
        for i in range(1, len(list)):
            # list[len(list) - i - 1][2] = list[len(list) - i][2]
            list[len(list) - i - 1][2] = list[len(list) - i][2] / ((len(list) - 1) ** (1/(len(list) - 1)))
        # print(list)
        for i in range(len(list)):
            agent.remember(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4])
        # 确保有足够的经验进行回放
        agent.timesteps_since_update += len(list)
        if agent.timesteps_since_update % agent.update_frequency == 0:
            agent.replay(agent.batch_size)
            agent.timesteps_since_update = 0
            agent.update_epsilon()
        if (reward > 0):
            wins += 1
        if reward == 0:
            wins += 0.5
        totalrew += reward
        if (e % 20 == 0):
            # with open('training_episode_log2.txt', 'a') as f:
            #     with contextlib.redirect_stdout(f):
            print(f"Episode {e + 1}/{episodes} completed")
            print(f"winrate: {wins}/{e+1}={wins/(e+1)}")
            print(f"avg rewards: {totalrew}/{e+1}={totalrew/(e+1)}")
        if (e % 1000 == 0):
            custom_agent.model.save('saved_model/custom_dqn_model13.h5')
            agent_dict = custom_agent.__dict__.copy()
            if 'model' in agent_dict:
                del agent_dict['model']

            # 保存代理的状态
            with open('saved_model/custom_dqn_agent13.pkl', 'wb') as f:
                pickle.dump(agent_dict, f)

def build_custom_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(64, input_dim=state_size, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

class CustomDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, batch_size):
        super().__init__(state_size, action_size, batch_size)
        self.model = None # build_custom_model(state_size, action_size)
# 初始化环境和代理
# agent = DQNAgent(state_size=(12,), action_size=12, batch_size=32)  # 输入为12种牌的数量，输出为摸牌的选择
# 创建一个新的DQNAgent实例
# new_agent = DQNAgent(state_size=(12,), action_size=12, batch_size=128)
custom_agent = DQNAgent(state_size=(12,), action_size=12, batch_size=256)
custom_agent.model = loaded_model
custom_agent.__dict__.update(loaded_agent_dict)
oppo_agent = DQNAgent(state_size=(12,), action_size=12, batch_size=256)

# new_agent.model = build_model(new_agent.state_size, new_agent.action_size)
# new_agent.__dict__.update(loaded_agent_dict)
# new_agent.model = loaded_model  # 将加载的模型赋给新实例
oppo_agent.__dict__.update(loaded_agent_dict2)
oppo_agent.model = loaded_model2  # 将加载的模型赋给新实例
# 开始训练
train_agent(custom_agent,oppo_agent, episodes=10000)
custom_agent.model.save('saved_model/custom_dqn_model13.h5')
agent_dict = custom_agent.__dict__.copy()
if 'model' in agent_dict:
    del agent_dict['model']

# 保存代理的状态
with open('saved_model/custom_dqn_agent13.pkl', 'wb') as f:
    pickle.dump(agent_dict, f)