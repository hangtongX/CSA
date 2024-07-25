import os
from pathlib import Path

import munch
import pandas as pd
import torch
from colorama import Fore, init as colorama_init
import numpy as np
from eval import metrics
from utils.getters import get_dataset
from styleframe import StyleFrame

colorama_init(autoreset=True)


class Evaluator:
    def __init__(self, config: munch.Munch):
        self.sortType = None
        self.patience_max = None
        self.patience_counter = 0
        self.evalmetric = None
        self.evalatk = None
        self.type = None
        self.testmetric = None
        self.testmetrick = None
        self.update(config.test)
        self.device = (
            str('cuda:' + str(config.train.device))
            if torch.cuda.is_available() and not config.train.no_cuda
            else "cpu"
        )
        self.val_best = np.inf if self.sortType == 'desc' else -np.inf
        # self.evaset, self.testset = get_dataset(config.dataset.name, config)

    def update(self, params: dict):
        self.__dict__.update(params)

    def early_stop_controller(self, performance):
        self.patience_counter += 1
        update_flag = False

        if self.sortType == "desc":
            if performance < self.val_best - 1e-6:
                self.val_best = performance
                self.patience_counter = 0
                update_flag = True

        elif self.sortType == "asc":
            if performance > self.val_best + 1e-6:
                self.val_best = performance
                self.patience_counter = 0
                update_flag = True
        else:
            raise Exception(Fore.RED + "Invalid sort type, please check...............")

        if self.patience_counter >= self.patience_max:
            return True, update_flag, self.val_best
        return False, update_flag, self.val_best

    def reset_patience(self):
        self.patience_counter = 0
        self.val_best = np.inf if self.sortType == 'desc' else -np.inf

    def eval_controller(self, score, **kwargs):
        if self.type == 'samplesort':
            score = score.unsqueeze(0)
            pos_matrix = torch.zeros_like(score, dtype=torch.int, device=self.device)
            _, topk_idx = torch.topk(score, max(self.testmetrick))
            pos_matrix[:, torch.zeros(1, dtype=torch.int).to(self.device)] = 1
            pos_len_list = pos_matrix.sum(dim=-1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=-1, index=topk_idx)
        elif self.type == 'fullsort':
            if (
                kwargs['mask'] is None
                or kwargs['ground_truth'] is None
                or kwargs['id'] is None
            ):
                raise NotImplementedError(
                    Fore.RED
                    + "No mask or ground truth params which is needed in fullsort eval"
                )
            mask = kwargs['mask']
            ground_truth = kwargs['ground_truth']
            pos_matrix = torch.zeros_like(score, dtype=torch.int, device=self.device)
            ids = (
                torch.arange(0, kwargs['id'].shape[0], 1)
                .to(score.device)
                .unsqueeze(1)
                .cpu()
                .numpy()
                .tolist()
            )
            ids_ = kwargs['id'].cpu().numpy().tolist()
            score[ids, mask[ids_]] = -np.inf
            pos_matrix[ids, ground_truth[ids_].tolist()] = 1
            _, topk_idx = torch.topk(score, self.evalatk, dim=1)
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
        else:
            raise NotImplementedError(
                Fore.RED + 'Unrecognized eval metric type........'
            )

        eval_func = self._get_metrics(self.evalmetric)
        rates = eval_func(topk_idx=topk_idx, pos_idx=pos_idx, pos_len=pos_len_list)
        return rates[:, self.evalatk - 1].mean()

    def test_controller(self, score, **kwargs):
        test_result = []
        if self.type == 'samplesort':
            score = score.unsqueeze(0)
            pos_matrix = torch.zeros_like(score, dtype=torch.int, device=self.device)
            _, topk_idx = torch.topk(score, max(self.testmetrick))
            pos_matrix[:, torch.zeros(1, dtype=torch.int).to(self.device)] = 1
            pos_len_list = pos_matrix.sum(dim=-1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=-1, index=topk_idx)
        elif self.type == 'fullsort':
            if (
                kwargs['mask'] is None
                or kwargs['ground_truth'] is None
                or kwargs['id'] is None
            ):
                raise NotImplementedError(
                    Fore.RED
                    + "No mask or ground truth params which is needed in fullsort eval"
                )
            mask = kwargs['mask']
            ground_truth = kwargs['ground_truth']
            pos_matrix = torch.zeros_like(score, dtype=torch.int, device=self.device)
            ids = (
                torch.arange(0, kwargs['id'].shape[0], 1)
                .to(score.device)
                .unsqueeze(1)
                .cpu()
                .numpy()
                .tolist()
            )
            ids_ = kwargs['id'].cpu().numpy().tolist()
            score[ids, mask[ids_, :].tolist()] = -np.inf
            pos_matrix[ids, ground_truth[ids_, :].tolist()] = 1
            _, topk_idx = torch.topk(score, max(self.testmetrick), dim=-1)
            pos_len_list = pos_matrix.sum(dim=-1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=-1, index=topk_idx)
        else:
            raise NotImplementedError(
                Fore.RED + 'Unrecognized eval metric type........'
            )
        for metric in self.testmetric:
            result = []
            test_func = self._get_metrics(metric)
            rates = test_func(
                topk_idx=topk_idx,
                pos_idx=pos_idx,
                pos_len=pos_len_list,
                item_pops=torch.FloatTensor(kwargs['item_popularity']).to(self.device),
            )
            for k in self.testmetrick:
                result.append(rates[:, k - 1].mean().item())
            test_result.append(result)
        return test_result

    def _get_metrics(self, metricname):
        if isinstance(metricname, list):
            for name in metricname:
                try:
                    metric = getattr(metrics, name.lower())
                    return metric
                except AttributeError as e:
                    print(
                        Fore.RED
                        + f'metric {name} in {metricname} is invalid, please check.......'
                    )
        elif isinstance(metricname, str):
            try:
                metric = getattr(metrics, metricname.lower())
                return metric
            except AttributeError as e:
                print(Fore.RED + f'metric {metricname} is invalid, please check.......')
        else:
            raise NotImplementedError(
                Fore.RED + 'Unrecognized eval metric, must be list or str.......'
            )

    def predict(self, model, data):
        if self.sortType == 'fullsort':
            pass
        elif self.sortType == 'samplesort':
            pass
        else:
            raise NotImplementedError(
                Fore.RED + 'Sort type not implement, please check.......'
            )

        return None

    def save_result_to_excel(self, data, path=None, hyper_turn=False):
        if path is None:
            dir_path = Path(os.path.abspath(__file__)).resolve().parents[-4]
            path = os.path.join(dir_path, "model_results/", 'project_no_name')
        for id, k in enumerate(self.topk):
            for metric in self.metrics:
                res = format(np.mean(self.result[id][metric]), '.6f')
                self.result[id][metric] = res
        back = (
            pd.DataFrame(self.result)
            .T.rename(columns={i: self.filename for i, k in enumerate(self.topk)})
            .T.reset_index()
        )
        if not os.path.exists(self.save_path):
            pd.DataFrame(columns=self.metrics).to_excel(
                self.save_path, f'K = {self.topk[0]}'
            )
            with pd.ExcelWriter(self.save_path, engine='openpyxl', mode='a') as writer:
                for i, k in enumerate(self.topk):
                    if i != 0:
                        pd.DataFrame(columns=self.metrics).to_excel(writer, f'K = {k}')
        # book = load_workbook(self.save_path)
        start = pd.read_excel(
            self.save_path, sheet_name=f'K = {k}', engine='openpyxl'
        ).shape[0]
        with pd.ExcelWriter(
            self.save_path, engine='openpyxl', mode='a', if_sheet_exists='overlay'
        ) as writer:
            # writer.book = book
            # writer.sheets.update( {sheet.title: sheet for sheet in book.worksheets})
            for i, k in enumerate(self.topk):
                back[i : i + 1].to_excel(
                    writer, f'K = {k}', startrow=start + 1, header=None, index=None
                )
        pass
