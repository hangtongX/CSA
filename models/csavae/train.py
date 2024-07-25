import logging
import os.path
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib.patheffects as Patheffects
from sklearn import manifold
import numpy as np
import torch.distributed as dist
import munch
import pandas as pd
import torch
from colorama import Fore, init as colorama_init
from utils.plots import plot_causal_graph
from trainer.vae_trainer import Trainer as BaseVAETrainer

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)
colorama_init(autoreset=True)


class Trainer(BaseVAETrainer):
    def __init__(
        self,
        modelname: str,
        dataname: str,
        config: munch.Munch = None,
    ):
        super(Trainer, self).__init__(modelname, dataname, config)

    def extra_output(self):
        out_dir = super().extra_output()
        graph = self.model.get_graph().cpu().detach().numpy()
        out_dir += (
            '/alpha_'
            + str(self.model.alpha)
            + '_concepts_k'
            + str(self.model.concepts_k)
        )
        self.t_sne()
        plot_causal_graph(graph, '/home/hangtong/codes/plot_figures/csc/')

    def t_sne(self):
        confounders = None
        pref_u = None
        for inputs in self.test_loader:
            inputs = self._set_inputs_to_device(inputs)
            with torch.no_grad():
                model_output = self.model.full_sort(
                    inputs,
                    # dataset_size=len(self.test_loader.dataset),
                    uses_ddp=self.distributed,
                )
                if confounders is None:
                    confounders = model_output.confounders.cpu().numpy()
                    pref_u = model_output.pref_u.cpu().numpy()
                else:
                    confounders = np.concatenate(
                        (confounders, model_output.confounders.cpu().numpy())
                    )
                    pref_u = np.concatenate((pref_u, model_output.pref_u.cpu().numpy()))
        tsne = manifold.TSNE(
            n_components=2,
            init='pca',
            random_state=15,
            perplexity=50,
            learning_rate=500,
            n_iter=2000,
            early_exaggeration=3,
        )
        confounders_label = torch.arange(4).repeat(confounders.shape[0]).cpu().numpy()
        confounders = confounders.reshape(-1, confounders.shape[2])
        confounders = np.concatenate((confounders, pref_u))
        confounders_label = np.concatenate(
            (
                confounders_label,
                torch.arange(1).repeat(pref_u.shape[0]).cpu().numpy() + 4,
            )
        )
        x_tsne = tsne.fit_transform(confounders)

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 16
        fig = plt.figure(figsize=(8, 8))
        for i in range(5):
            plt.scatter(
                x_tsne[confounders_label == i, 0],
                x_tsne[confounders_label == i, 1],
                # label=labels[i],
                marker='o',
                s=1,
            )
            xtext, ytext = (
                np.median(x_tsne[confounders_label == i, :], axis=0)[0],
                np.min(x_tsne[confounders_label == i, :], axis=0)[1] - 5,
            )
            if i != 4:
                txt = plt.text(
                    xtext, ytext, '$\epsilon_{}$'.format(str(i)), fontsize=20
                )
            else:
                txt = plt.text(xtext, ytext, 'user', fontsize=20)
            txt.set_path_effects(
                [
                    Patheffects.Stroke(linewidth=5, foreground="w"),
                    Patheffects.Normal(),
                ]
            )

        # plt.legend(loc=1)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        # plt.title('Visualization of confounders and user preference')
        plt.savefig('/home/hangtong/codes/plot_figures/csc/figure9_1.pdf', dpi=300)
        plt.show()
