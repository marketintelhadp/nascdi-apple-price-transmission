import numpy as np
import matplotlib.pyplot as plt

h = np.arange(0, 12)
pos = 0.25 * (1 - np.exp(-0.4 * h))
neg = 0.45 * (1 - np.exp(-0.6 * h))

plt.figure(figsize=(6, 4))
plt.plot(h, pos, label="Positive shock")
plt.plot(h, neg, label="Negative shock", linestyle="--")

plt.xlabel("Weeks after shock")
plt.ylabel("Cumulative price response")
plt.title("Stylized dynamic adjustment paths")
plt.legend()
plt.tight_layout()
plt.savefig("results/nardl/plots/fig7_dynamic_multipliers.png", dpi=300)
plt.close()
