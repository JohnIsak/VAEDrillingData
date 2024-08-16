import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def plot_training(train, val, model):
    data = pd.DataFrame({"Train": train, "Val": val})
    ax = sb.lineplot(data=data)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title(model.name + " loss")
    plt.savefig("plots/"+model.name+"_loss.pdf")
    plt.show()