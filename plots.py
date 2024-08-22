import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pickle
import torch.nn as nn
import numpy as np
import torch

#WOB INDEX 1 RPM INDEX 5

row_labels = [
    "Measured Depth", "Weight on Bit", "Pressure", "Torque", "Rate of Penetration",
    "RPM", "Mud Flow", "Mud Density", "Diameter",
    "Hook Load", "True Vertical Depth", "Gamma"
]
units = ["m", "tf", "kPa", "kN.m", "m/h", "rpm", "L/min", "g/cm3", "mm", "tf", "m", "gAPI"]

# row_labels_2 = [
#    "Measured Depth", "Pressure", "Torque", "Rate of Penetration", "Mud Flow", "Mud Density", "Diameter",
#    "Hook Load", "True Vertical Depth", "Gamma"]
# units_2 = ["m", "kPa", "kN.m", "m/h", "L/min", "g/cm3", "mm", "tf", "m", "gAPI"]
plt.rcParams['font.family'] = 'serif'

# asd = pd.read_csv("data/USROP_A 0 N-NA_F-9_Ad.csv")
# print(asd.columns)

indices_non_control_parameters = torch.tensor([0, 2, 3, 4, 8, 9, 10, 11])


def index_select(input_list, indices):
    return [input_list[i] for i in indices]

def plot_training(train, val, model):
    data = pd.DataFrame({"Train": train, "Val": val})
    ax = sb.lineplot(data=data)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title(model.name + " loss")
    plt.savefig("plots/"+model.name+"_loss.pdf")
    plt.show()

def plot_anomaly(data, indexes):

    data = data.dataset.X.cpu().numpy()
    tranformer = pickle.load(open("models/data_transform.pkl", "rb"))
    # indexes = np.random.randint(0, len(data), 10)

    # USING REDUCED INDICIES
    global row_labels
    global units
    row_labels = index_select(row_labels, indices_non_control_parameters)
    units = index_select(units, indices_non_control_parameters)

    for index in indexes:
        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        axes = axes.flatten()
        tranformed = tranformer.inverse_transform(data[index, :])
        print(tranformed.shape, "shape")
        for i, label in enumerate(row_labels):
            # sb.lineplot(ax = axes[i], data=tranformed[:, i])
            sb.lineplot(ax=axes[i], x = np.arange(tranformed.shape[0])*5, y=tranformed[:, i])
            # sb.lineplot(ax=axes[i], data=tranformed)
            axes[i].set_xlabel("cm")
            axes[i].set_ylabel(units[i])
            axes[i].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            axes[i].set_title(label)
        # plt.title(f"Anomaly {index}")
        # plt.title(f"Random Sample {index}")
        plt.tight_layout()
        plt.savefig(f"plots/anomaly_{index}.pdf")
        plt.show()


# Currently does not include ROP
def plot(data, model: nn.Module, indexes):
    model.eval()
    # indexes = np.random.randint(0, len(data.dataset.X), 10)

    # USING REDUCED INDICIES
    global row_labels
    global units
    row_labels = index_select(row_labels, indices_non_control_parameters)
    units = index_select(units, indices_non_control_parameters)

    for index in indexes:
        x_recon, _, _  = model.forward(data.dataset.X[index].unsqueeze(0))
        x_recon = x_recon
        original = data.dataset.X[index]
        error = x_recon - original
        error = torch.index_select(error, 2, indices_non_control_parameters)
        error = error.cpu().detach().numpy()
        heatmap = np.zeros((len(units), 30))
        for i in range(30):
            heatmap[:, i] = error[:, i]
        plt.figure(figsize=(12, 8))
        sb.heatmap(heatmap, xticklabels=np.arange(original.shape[0])*5, yticklabels=row_labels)
        plt.xlabel("cm")
        plt.savefig(f"plots/heatmap_random_sample_{index}.pdf")
        plt.title(f"Random Sample {index}")
        plt.title(f"Anomaly {index}")
        plt.show()

