import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_actor(state_size, action_size):
    inputs = Input(shape=(state_size,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(action_size, activation='softmax')(x)  # 对于离散动作空间使用 softmax
    return Model(inputs, outputs)

def build_critic(state_size):
    inputs = Input(shape=(state_size,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)  # 输出状态价值
    return Model(inputs, outputs)

class PPOLiteAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lam=0.95, epochs=10, clip_ratio=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.clip_ratio = clip_ratio
        self.actor = build_actor(state_size, action_size)
        self.critic = build_critic(state_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train(self, states, actions, advantages, old_probs, returns):
        with tf.GradientTape() as tape:
            probs = self.actor(states)
            values = self.critic(states)
            
            # 计算新的动作概率
            new_probs = tf.gather_nd(probs, tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1))
            ratio = new_probs / (old_probs + 1e-10)
            
            # 计算剪裁的目标策略损失
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            
            # 计算值函数损失
            value_loss = tf.reduce_mean((returns - values) ** 2)
            
            loss = policy_loss + 0.5 * value_loss
            
        gradients = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables))

    def act(self, state):
        state = tf.expand_dims(state, 0)
        probs = self.actor(state)
        action = tf.random.categorical(tf.math.log(probs), num_samples=1)
        return int(action.numpy()[0][0])

    def get_advantages_and_returns(self, rewards, values, dones):
        advantages = []
        returns = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae + values[t])
            returns.insert(0, gae + values[t])
        return advantages, returns
    
def train_ppo_agent(agent, env, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_rewards = []
        episode_states = []
        episode_actions = []
        episode_probs = []
        episode_dones = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            episode_states.append(state)
            episode_actions.append(action)
            episode_probs.append(agent.actor(tf.expand_dims(state, 0)).numpy()[0][action])
            episode_dones.append(done)

            state = next_state

        values = agent.critic(tf.convert_to_tensor(episode_states, dtype=tf.float32))
        values = tf.concat([values, tf.zeros((1, 1))], axis=0)
        
        advantages, returns = agent.get_advantages_and_returns(episode_rewards, values, episode_dones)
        
        for _ in range(agent.epochs):
            agent.train(tf.convert_to_tensor(episode_states, dtype=tf.float32),
                        tf.convert_to_tensor(episode_actions, dtype=tf.int32),
                        tf.convert_to_tensor(advantages, dtype=tf.float32),
                        tf.convert_to_tensor(episode_probs, dtype=tf.float32),
                        tf.convert_to_tensor(returns, dtype=tf.float32))