import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# 保持现有的MahjongEnvironment类不变

class DQNAgent:
    # 保持DQNAgent类的实现不变
    # ...

class PPOAgent:
    def __init__(self, state_size, action_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = 0.99  # 折扣因子
        self.clip_ratio = 0.2  # PPO裁剪参数
        self.actor_lr = 0.0003
        self.critic_lr = 0.001
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.memory = []

    def _build_actor(self):
        # 构建策略网络（演员）
        actor = Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='softmax')
        ])
        actor.compile(optimizer=Adam(learning_rate=self.actor_lr))
        return actor

    def _build_critic(self):
        # 构建价值网络（评论家）
        critic = Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        critic.compile(optimizer=Adam(learning_rate=self.critic_lr), loss='mse')
        return critic

    def act(self, state):
        # 根据当前状态选择动作
        probs = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=probs)
        return action, probs[action]

    def remember(self, state, action, action_prob, reward, next_state, done):
        # 存储经验
        self.memory.append((state, action, action_prob, reward, next_state, done))

    def train(self):
        # 训练PPO智能体
        if len(self.memory) < self.batch_size:
            return

        # 从记忆中采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, old_probs, rewards, next_states, dones = zip(*batch)

        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.array(actions)
        old_probs = np.array(old_probs)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # 计算优势和回报
        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        returns = advantages + values

        # 训练演员（策略网络）
        with tf.GradientTape() as tape:
            current_probs = self.actor(states)
            current_probs = tf.gather_nd(current_probs, tf.stack([tf.range(len(actions)), actions], axis=1))
            ratios = current_probs / old_probs
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -tf.reduce_mean(tf.minimum(ratios * advantages, clipped_ratios * advantages))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # 训练评论家（价值网络）
        self.critic.fit(states, returns, epochs=1, verbose=0)

        # 清空记忆
        self.memory.clear()

def train_agents(episodes, batch_size):
    env = MahjongEnvironment()
    dqn_agent = DQNAgent(12, 12, batch_size)
    ppo_agent = PPOAgent(12, 12, batch_size)
    
    for e in range(episodes):
        state1, state2 = env.reset()
        state1 = np.reshape(state1, [1, 12])
        state2 = np.reshape(state2, [1, 12])
        
        total_reward_dqn = 0
        total_reward_ppo = 0
        
        for time in range(500):  # 设置最大步数
            # DQN智能体选择动作
            action1 = dqn_agent.act(state1)
            
            # PPO智能体选择动作
            action2, action_prob2 = ppo_agent.act(state2)
            
            # 环境步进
            next_state1, reward1, done, _ = env.step(action1, action2)
            next_state1 = np.reshape(next_state1, [1, 12])
            next_state2 = np.reshape(env.opponentState, [1, 12])
            
            # 记录经验
            dqn_agent.remember(state1, action1, reward1, next_state1, done)
            ppo_agent.remember(state2, action2, action_prob2, -reward1, next_state2, done)
            
            total_reward_dqn += reward1
            total_reward_ppo -= reward1
            
            state1 = next_state1
            state2 = next_state2
            
            if done:
                break
        
        # 训练智能体
        if len(dqn_agent.memory) > batch_size:
            dqn_agent.replay(batch_size)
        ppo_agent.train()
        
        # 打印训练进度
        print(f"Episode {e}/{episodes}, DQN Reward: {total_reward_dqn}, PPO Reward: {total_reward_ppo}")
        
        # 逐步降低DQN的探索率
        if dqn_agent.epsilon > dqn_agent.epsilon_min:
            dqn_agent.epsilon *= dqn_agent.epsilon_decay

    return dqn_agent, ppo_agent

# 训练模型
trained_dqn_agent, trained_ppo_agent = train_agents(episodes=10000, batch_size=32)

# 保存模型
trained_dqn_agent.model.save('saved_model/dqn_model.h5')
trained_ppo_agent.actor.save('saved_model/ppo_actor_model.h5')
trained_ppo_agent.critic.save('saved_model/ppo_critic_model.h5')