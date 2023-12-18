import torch

import matplotlib.pyplot as plt


steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
sigmas = [80.0, 57.6, 40.8, 28.4, 19.4, 12.9, 8.4, 5.3, 3.3, 1.1, 0.3]

diffs_gauss = []
diffs_iso = []

for step in steps:
    score_gaussian = torch.load(f"scores/gaussian/{step}.pt").flatten(1)
    score_neural = torch.load(f"scores/neural/{step}.pt").flatten(1)
    score_iso = torch.load(f"scores/isotropic/{step}.pt").flatten(1)

    diffs_gauss.append((torch.norm(score_gaussian - score_neural, dim=1) / torch.norm(score_neural, dim=1)).mean())
    diffs_iso.append((torch.norm(score_iso - score_neural, dim=1) / torch.norm(score_neural, dim=1)).mean())

plt.plot(sigmas, diffs_gauss, label="Gaussian score")
plt.plot(sigmas, diffs_iso, label="Isotropic score")
plt.xlabel(r"Noise scale $\sigma_t$")
plt.ylabel("Relative score difference")
plt.title("Score field approximation error")
plt.yscale("log")
plt.legend()
plt.savefig(f"scores/scores.png");

print("Saved to 'scores/scores.png'")

