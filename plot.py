import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

with open("adanet_mnist_adaptiveB_noregularize.log", "r") as f:
    data = f.readlines()

epoch_losses = [
    float(l.strip().split(" ")[-1])
    for l in data
    if "Train" in l and "Loss" in l and "Candid" not in l
]

net_losses = [float(l.strip().split(" ")[-4]) for l in data if "Candid" in l]

for i, l in enumerate(net_losses):
    if i % 2 == 0:
        if l == epoch_losses[i // 2 + 1]:
            sns.lineplot(
                [i // 2 + 1, i // 2], [l, epoch_losses[i // 2]], color="#4c72b0",
            )
        else:
            sns.lineplot(
                [i // 2 + 1, i // 2],
                [l, epoch_losses[i // 2]],
                dashes=[(2, 0)],
                color="#4c72b0",
                marker="s",
            )
    else:
        if l == epoch_losses[i // 2 + 1]:
            sns.lineplot(
                [i // 2 + 1, i // 2], [l, epoch_losses[i // 2]], color="#dd8452",
            )
        else:
            sns.lineplot(
                [i // 2 + 1, i // 2],
                [l, epoch_losses[i // 2]],
                color="#dd8452",
                marker="s",
            )


sns.despine()

plt.title(r"Subnetwork Selection: Training Loss vs Epoch ($\gamma = 0$)")
plt.ylabel("Training Loss")
plt.xlabel("Epoch")

plt.savefig("plot.png", dpi=128)
