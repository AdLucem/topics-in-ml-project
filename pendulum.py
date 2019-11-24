"""
Code taken from the following source, and altered:
__name__   = predict.py
__author__ = Yash Patel
__description__ = Full prediction code of OpenAI Cartpole environment using Keras
"""

import gym
import numpy as np

from model import create_model
from data import gather_data


def predict():

    env = gym.make("Pendulum-v0")

    trainingX, trainingY = gather_data(env)

    model = create_model()
    # change the STATE of the MODEL yes
    # TRAIN IT LIKE THAT WHY NOT
    model.fit(trainingX, trainingY, epochs=5)

    scores = []
    num_trials = 50
    sim_steps = 500
    for _ in range(num_trials):
        observation = env.reset()
        score = 0
        for step in range(sim_steps):
            action = np.argmax(model.predict(observation.reshape(1, 4)))
            observation, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)

    print(np.mean(scores))


if __name__ == "__main__":
    predict()