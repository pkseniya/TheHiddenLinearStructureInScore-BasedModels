import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("images_with_skips").absolute()
SAVE_PATH = Path("experiments_with_skip_results.txt").absolute()
PLOT_PATH = Path("experiments_with_skip_graph.png").absolute()


results = {}
for file in RESULTS_DIR.iterdir():
    if not str(file).endswith(".txt"):
        continue
    n_skip = str(file).split("/")[-1].split(".")[0]
    with open(file) as file:
        for line in file:
            pass
    score = float(line[:-1])
    results[n_skip] = score

with open(SAVE_PATH, "w") as file:
    file.write("n_skip  fid\n")
    file.write("-------------\n")
    for n_skip in sorted(map(int, results)):
        file.write(f"{n_skip:<7} {np.round(results[str(n_skip)], 3)}\n")
    

n_skips = sorted(map(int, results))
scores = [results[str(n_skip)] for n_skip in n_skips]
plt.plot(n_skips, scores, "-o")
plt.title("Experiment with skipping steps\ndataset=CIFAR10 -- n_steps=18 -- solver='heun'")
plt.xlabel("#(skipped steps)")
plt.ylabel("FID")
plt.savefig(PLOT_PATH)

print("outputs:")
print(str(SAVE_PATH))
print(str(PLOT_PATH))

