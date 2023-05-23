import matplotlib.pyplot as plt
import numpy as np

from channels.quadriga import QuadrigaChannel

max_number_ues = 100
max_number_basestations = 1
num_available_rbs = np.array([135])
channel_gen = QuadrigaChannel(
    max_number_ues, max_number_basestations, num_available_rbs
)
episode = 0
steps_per_episode = 1000

average_se = np.zeros((steps_per_episode, max_number_ues))

for step in range(steps_per_episode):
    spectral_eff = channel_gen.step(step, episode, np.array([1]))
    average_se[step, :] = np.mean(np.squeeze(spectral_eff), axis=1)

plt.figure()
for ue in range(max_number_ues):
    plt.plot(np.arange(steps_per_episode), average_se[:, ue], label=f"UE {ue}")
plt.ylim([0, 3])
plt.xlabel("Step")
plt.ylabel("Average Spectral Efficiency")
plt.show()
