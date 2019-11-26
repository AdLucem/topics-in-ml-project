import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque
import matplotlib.pyplot as plt 

# determines how to assign values to each state
# i.e. takes the state
# and action (two-input model) and
# determines the corresponding value

def pad(arr, size):
    original_size = arr.shape[1]
    padding_size = size - original_size
    zeroes = np.zeros([1, padding_size])
    return np.concatenate([arr, zeroes], axis=1)

def unpad(arr, size):
    return arr[0:, 0:size]


class ActorCritic:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95

        # ============================================ #
        #                Actor Model                   #
        # Chain rule: find the gradient of chaging the
        # actor network params in getting closest to the
        # final value network predictions, i.e. de/dA
        # Calculate de/dA as = de/dC * dC/dA, where e is
        # error, C critic, A act
        # ============================================= #

        self.memory = deque(maxlen=2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(
            tf.float32,
            [None, self.env.observation_space.shape[0]])
        # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(
            self.actor_model.output,
            actor_model_weights,
            -self.actor_critic_grad)
        # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).apply_gradients(grads)

        # ================================================ #
        #                Critic Model                      #
        # ================================================ #

        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)
        # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.global_variables_initializer())

    # ============================================== #
    #             Model Definitions                  #
    # ============================================== #

    def create_actor_model(self):

        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24, activation='tanh')(state_input)
        h2 = Dense(24, activation='tanh')(h1)
        h3 = Dense(24, activation='tanh')(h2)
        output = Dense(self.env.observation_space.shape[0], activation='tanh')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape,
                            name="state_input")
        state_h1 = Dense(24, activation='relu',
                         name="state_h1")(state_input)

        action_input = Input(shape=self.env.observation_space.shape,
                             name="action_input")
        action_h1 = Dense(24, name="action_h1")(action_input)

        merged = Add(name="merged")([state_h1, action_h1])
        merged_h1 = Dense(24, activation='relu',
                          name="merged_h1")(merged)
        output = Dense(1, activation='relu',
                       name="output")(merged_h1)

        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ================================================== #
    #            Model Training                          #
    # ================================================== #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        # print("Training actor network using:")
        # print("(A) Our sample set")
        # print("(B) The action predicted by current actor network, given the action from our sample set")
        # print("(C) The critic network's scoring of our current actor model's predicted action")
        # print("The gradient backpropped onto the actor network is:")
        # print("dE/dA = dE/dC * dC/dA")
        # print("E: error between critic's scoring of episode-sample action and predicted action")
        # print("C: critic network weights")
        # print("A: actor network weights")
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)

            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        # print("Training critic network using our current (Si, Ai) -> Ri pairs...")
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                target_action_unpad = unpad(target_action, 4) 
                avg_action = np.mean(target_action_unpad)

                ###########################################
                future_reward = 0
                for i in range(len(target_action_unpad)):
                    temp = target_action[i]
                    target_action[i] = avg_action
                    future_reward += self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                    target_action[i] = temp
                ###########################################

                reward += self.gamma * future_reward

            cur_state = cur_state.reshape((1, 24))
            action = action.reshape((1, 24))
            reward = np.array(reward)
            reward = reward.reshape((1, 1))

            self.critic_model.fit([cur_state, action], reward, verbose=0)
            return reward

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return False, "error"

        # print("Since we've collected batch_size=" + str(batch_size) + " samples,")
        # print("We train the actor-critic network on one batch.")
        # print("--------------------------------------------------")
        samples = random.sample(self.memory, batch_size)
        reward = self._train_critic(samples)
        self._train_actor(samples)
        return True, reward
        # print("==================================================")

    # =================================================== #
    #            Target Model Updating                    #
    # =================================================== #

    def _update_actor_target(self):
        # print("Updating actor...")
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)-2):
            actor_target_weights[i] = \
            actor_model_weights[i].reshape(actor_target_weights[i].shape)
        
        actor_target_weights[6] = \
        np.mean(actor_model_weights[6], axis=0).reshape(actor_target_weights[6].shape)
        actor_target_weights[7] = \
        np.mean(actor_model_weights[7], axis=0).reshape(actor_target_weights[7].shape)

        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        # print("Updating critic...")
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        # print("Updating the target functions...")
        # print("----------------------------------")
        self._update_actor_target()
        self._update_critic_target()
        # print("==================================")

    # ================================================= #
    #               Model Predictions                   #
    # ================================================= #

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return pad(self.env.action_space.sample().reshape(1, 4), 24)
        return self.actor_model.predict(cur_state)


def main():
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make("BipedalWalker-v2")
    actor_critic = ActorCritic(env, sess)

    NUM_ITERATIONS = 10

    episode = 0
    epochs = 0
    episode_rewards = []
    max_epochal_rewards = []
    max_epochal_scores = []

    for i in range(NUM_ITERATIONS):
        print("Episode ", episode)
        cur_state = env.reset()
        action = env.action_space.sample()
        done = False
        epoch = 0
        updated = False
        cum_reward_epoch = 0
        cum_reward_episode = 0
        epochal_rewards = []
        epochal_scores = []

        while not done:

            if updated:
                print("Epoch ", epoch, "with reward ", cum_reward_epoch)
                cum_reward_episode += cum_reward_epoch
                epochal_rewards.append(cum_reward_epoch)
                epoch += 1
                cum_reward_epoch = 0
                updated = False

            env.render()
            cur_state = cur_state.reshape((1, env.observation_space.shape[0]))

            action = actor_critic.act(cur_state)
            action_taken = unpad(action, 4).reshape((4))
            action_keras = action.reshape((1, env.observation_space.shape[0]))

            new_state, reward, done, _ = env.step(action_taken)
            cum_reward_epoch += reward
            new_state = new_state.reshape((1, env.observation_space.shape[0]))

            actor_critic.remember(cur_state, action_keras, reward, new_state, done)
            trained, score = actor_critic.train()
            if trained:
                actor_critic.update_target()
                actor_critic.memory = []
                updated = True
                epochal_scores.append(score[0][0])

            cur_state = new_state

        episode += 1
        epochs += epoch
        episode_rewards.append(cum_reward_episode)
        max_epochal_rewards.append(max(epochal_rewards))
        max_epochal_scores.append(max(epochal_scores))

    return episode, epochs, episode_rewards, max_epochal_rewards, max_epochal_scores


if __name__ == "__main__":
    x1, x2, y1, y2, y3 = main()

    plt.subplot(2, 1, 1)
    plt.plot(range(1, x1+1), y1)
    plt.ylabel('Reward per episode')

    #plt.subplot(2, 1, 2)
    #plt.plot(range(1, x1+1), y2)
    #plt.ylabel('Max. reward per epoch')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, x1+1), y3)
    plt.ylabel('Max. score per epoch')

    plt.savefig("increased_lr_avg_action_10_iters.png")
    plt.show()
