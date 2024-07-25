import os

import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm
from colorama import Fore, init as colorinit
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from data_utils.dataset_config import DatasetConfig
from data_utils.load_data import DataLoad
from easydict import EasyDict
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed

colorinit(autoreset=True)


class BaseDataset(Dataset):
    """
    Base class for datasets used, return with data and label
    """

    def __init__(self, data, configs: DatasetConfig = None, *args, **kwargs):
        super(BaseDataset, self).__init__()
        self.data = data
        self.label = configs.label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.label is not None:
            sample = {'data': self.data[index][0], 'label': self.data[index][1]}
        else:
            sample = {'data': self.data[index][0]}
        return sample


def get_dataclass(type: str):
    dataclasses = {
        'normal': BaseDataset,
        'rec': RecDataset,
        'rec_neg': RecNegativeDataset,
        'rec_pairneg': RecPairDataset,
        'rec_pair_explain': RecPairExplainDataset,
        'rec_vae': RecVaeDataset,
        'rec_sample': RecSampleForEvalDataset,
        'rec_full': RecFullSortForTestDataset,
    }
    if type in dataclasses:
        return dataclasses[type]
    else:
        raise NotImplementedError(type)


def generate_dataset(dataconfig: dict):
    config = DatasetConfig()
    config.update(dataconfig)
    load_data = DataLoad(config)
    datainfo = load_data.load_static()
    data = load_data.load()
    negatives = load_data.load_item_pool()
    trainset = get_dataclass(config.traintype)(
        data=data['train'].copy().astype(float).to_numpy(),
        configs=config,
        negatives=negatives,
        total_data=data.copy(),
        dataInfo=datainfo,
    )
    testset = get_dataclass(config.testtype)(
        data['test'].to_numpy().copy().astype(float),
        config,
        negatives=negatives,
        total_data=data.copy(),
        dataInfo=datainfo,
    )
    evalset = get_dataclass(config.evaltype)(
        data['val'].to_numpy().copy().astype(float),
        config,
        negatives=negatives,
        total_data=data.copy(),
        dataInfo=datainfo,
    )
    return trainset, evalset, testset, datainfo


class RecDataset(BaseDataset):
    """
    Dataset for rec normal dataset, return with userid, itemid and label
    """

    def __init__(
        self,
        data,
        configs: DatasetConfig = None,
        binary=True,
        train_mask=True,
        *args,
        **kwargs,
    ):
        data = np.ceil(data)
        super(RecDataset, self).__init__(data=data, configs=configs, *args, **kwargs)
        if binary:
            self.binary_scores()
        self.dataInfo = kwargs['dataInfo']
        if train_mask:
            self.mask = self.generate_mask()
        self.ips_score = None
        self.train_data = kwargs['total_data']['train'].copy().to_numpy()
        self.calculate_ips_score()

    def __getitem__(self, index):

        if self.label:
            return {
                'user': self.data[index][0],
                'item': self.data[index][1],
                'label': self.data[index][2],
                'index': index,
                'mask': self.mask[index],
                'ips_score': (
                    self.ips_score.negative[int(self.data[index][1])]
                    if self.data[index][2] == 0
                    else self.ips_score.positive[int(self.data[index][1])]
                ),
            }
        else:
            return {
                'user': self.data[index][0],
                'item': self.data[index][1],
                'index': index,
                'mask': self.mask[index],
                'ips_score': (
                    self.ips_score.negative[int(self.data[index][1])]
                    if self.data[index][2] == 0
                    else self.ips_score.positive[int(self.data[index][1])]
                ),
            }

    def __calculate_item_popularity__(self, data=None, **kwargs):
        if data is None:
            data = pd.DataFrame(self.data[:, :2], columns=['user', 'item'])
        else:
            data = pd.DataFrame(data[:, :2], columns=['user', 'item'])
        item_count = data.groupby('item')['user'].count()
        item_count.colums = pd.Series(['count'])

        # set 1 for non-interaction items for easy to calculate
        pop = np.ones(self.dataInfo['item'][0])
        pop[item_count.index.astype(np.int32)] = item_count.values
        return pop

    def binary_scores(self):
        print(Fore.YELLOW + 'Binary scores to [0, 1]')
        score_max = self.data[:, 2].max()
        self.data[np.where(self.data[:, 2] * 10 < 8 * score_max), 2] = 0
        self.data[np.where(self.data[:, 2] * 10 >= 8 * score_max), 2] = 1

    def generate_mask(self):
        pop = self.__calculate_item_popularity__()
        min_popularity = min(pop)
        p = [min_popularity / pop[int(x)] + 1e-4 for x in self.data[:, 1]]
        p /= sum(p)
        index = np.random.choice(
            range(self.data.shape[0]), int(self.data.shape[0] / 7), p=p
        )
        mask = np.full(self.data.shape[0], False, dtype=bool)
        mask[index] = True

        return mask

    def calculate_ips_score(self):
        item_pop = self.__calculate_item_popularity__(self.train_data)
        max_count = item_pop.max()
        positive = np.power(item_pop / max_count, 0.5)
        negative = np.power(1 - item_pop / max_count, 0.5)
        self.ips_score = EasyDict({'positive': positive, 'negative': negative})

    def cal_explain_matrix(self):
        user_num, item_num, train = (
            self.dataInfo['user'][0],
            self.dataInfo['item'][0],
            pd.DataFrame(self.train_data[:, :2], columns=['user', 'item']),
        )
        interaction_matrix = pd.crosstab(train.user, train.item)
        missing_columns = list(set(range(item_num)) - set(list(interaction_matrix)))
        missing_rows = list(set(range(user_num)) - set(interaction_matrix.index))
        for missing_column in missing_columns:
            interaction_matrix[missing_column] = [0] * len(interaction_matrix)
        for missing_row in missing_rows:
            interaction_matrix.loc[missing_row] = [0] * item_num
        interaction_matrix = np.array(
            interaction_matrix[list(range(item_num))].sort_index()
        )
        item_similarity_matrix = cosine_similarity(interaction_matrix.T)
        np.fill_diagonal(item_similarity_matrix, 0)
        neighborhood = [
            np.argpartition(row, -20)[-20:] for row in item_similarity_matrix
        ]
        explain_matrix = (
            np.array(
                [
                    [
                        sum(
                            [
                                interaction_matrix[user, neighbor]
                                for neighbor in neighborhood[item]
                            ]
                        )
                        for item in range(item_num)
                    ]
                    for user in range(user_num)
                ]
            )
            / 20
        )
        return explain_matrix

    def generate_graph(self):
        num_user, num_item = self.dataInfo['user'][0], self.dataInfo['item'][0]
        train = self.train_data.copy()
        adj_mat = sp.dok_matrix(
            (num_user + num_item, num_user + num_item), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        UserItemNet = csr_matrix(
            (train[:, 2], (train[:, 0], train[:, 1])), shape=(num_user, num_item)
        )
        R = UserItemNet.tolil()
        adj_mat[:num_user, num_user:] = R
        adj_mat[num_user:, :num_user] = R.T
        adj_mat = adj_mat.todok()
        # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        return norm_adj


class RecNegativeDataset(RecDataset):
    """
    Dataset for rec negative dataset, generate negative items and combine with orign data, return with userid, itemid
    """

    def __init__(self, data, configs: DatasetConfig = None, *args, **kwargs):
        if kwargs['negatives'] is None:
            raise ValueError(
                Fore.RED
                + 'Negatives is needed to generate negative dataset, can not be None, please provide negative data'
            )
        self.negatives = kwargs['negatives']
        self.negative_num = configs.negative_num
        self.concact = configs.concact
        self.train_data = kwargs['total_data']['train']
        # Filter out items with low ratings
        data[:, 2] = data[:, 2].astype(np.float16)
        data = data[np.where(data[:, 2] * 10 >= 8 * data[:, 2].max())]
        super(RecNegativeDataset, self).__init__(
            data=data, binary=False, configs=configs, **kwargs
        )
        # self.__post_init__()

    def __post_init__(self):
        self.__sample__()
        if self.concact:
            self.binary_scores()
        print(
            Fore.GREEN
            + f'After sampling {self.negative_num} negative samples, the dataset size is {self.__len__()}'
        )

    def __sample_v1__(self):
        """
        Sample {self.negative_num} of negatives for each user-item postive pair
        :return:
        :rtype:
        """
        backdata = pd.DataFrame(self.data, columns=['user', 'item', 'rating'])
        negatives = np.array(
            self.negatives.loc[backdata.user.tolist()]
            .negative_items.apply(
                lambda x: np.random.choice(
                    list(x), self.negative_num, replace=False
                ).tolist()
            )
            .tolist()
        ).flatten()
        backdata = backdata.reindex(
            backdata.index.repeat(self.negative_num)
        ).reset_index(drop=True)
        backdata['negative'] = pd.Series(negatives)
        backdata['rating'] = pd.Series(
            np.zeros((backdata.shape[0], 1), dtype=np.int32).flatten()
        )
        if self.concact:
            dataset = pd.DataFrame(self.data, columns=['user', 'item', 'rating'])
            dataset['flag'] = dataset.item
            dataset = dataset[['user', 'item', 'flag', 'rating']]
            backdata = backdata[['user', 'negative', 'item', 'rating']]
            backdata.columns = ['user', 'item', 'flag', 'rating']
            backdata = pd.concat([dataset, backdata])
            backdata.sort_values(
                by=['user', 'flag', 'rating', 'item'],
                inplace=True,
                ascending=[True, True, False, True],
            )
            backdata.reset_index(inplace=True, drop=True)
            self.data = backdata[['user', 'item', 'rating']].to_numpy()
        else:
            self.data = backdata[['user', 'item', 'negative']].to_numpy()

    def __sample__(self):
        """
        Sample {self.negative_num} of negatives for each user-item postive pair
        :return:
        :rtype:
        """
        backdata = pd.DataFrame(self.train_data, columns=['user', 'item', 'rating'])
        interact_status = (
            backdata.groupby('user')['item']
            .apply(set)
            .reset_index()
            .rename(columns={'item': 'postive_items'})
            .sort_values(by='user', ascending=True)
        )

        def pattch(negatives, interact_status):
            return np.random.choice(
                list(negatives - interact_status),
                len(interact_status),
                replace=True,
            ).tolist()

        negatives = Parallel(n_jobs=-1, backend='threading')(
            delayed(pattch)(
                self.negatives,
                row['postive_items'],
            )
            for idx, row in tqdm(
                iterable=interact_status.iterrows(),
                total=interact_status.shape[0],
                desc='Processing negatives:',
                ncols=80,
                colour='yellow',
            )
        )
        negatives = np.asarray([neg for negative in negatives for neg in negative])
        backdata = backdata.reindex(
            backdata.index.repeat(self.negative_num)
        ).reset_index(drop=True)
        backdata.sort_values(
            by=['user', 'item'],
            inplace=True,
            ascending=[True, True],
        )
        backdata['negative'] = pd.Series(negatives)
        backdata['rating'] = pd.Series(
            np.zeros((backdata.shape[0], 1), dtype=np.int32).flatten()
        )
        backdata.dropna(inplace=True)
        if self.concact:
            dataset = pd.DataFrame(self.data, columns=['user', 'item', 'rating'])
            dataset['flag'] = dataset.item
            dataset = dataset[['user', 'item', 'flag', 'rating']]
            backdata = backdata[['user', 'negative', 'item', 'rating']]
            backdata.columns = ['user', 'item', 'flag', 'rating']
            backdata = pd.concat([dataset, backdata])
            backdata.sort_values(
                by=['user', 'flag', 'rating', 'item'],
                inplace=True,
                ascending=[True, True, False, True],
            )
            backdata.reset_index(inplace=True, drop=True)
            self.data = backdata[['user', 'item', 'rating']].to_numpy()
        else:
            self.data = backdata[['user', 'item', 'negative']].to_numpy()
            self.data = self.data.astype(np.int32)

    def re_sample(self):
        self.__sample__()
        print(Fore.YELLOW + 'Resampling negatives successfully......')

    def __init_subclass__(cls, **kwargs):
        def init_decorator(previous_init):
            def new_init(self, *args, **kwargs):
                previous_init(self, *args, **kwargs)
                if isinstance(self, cls):
                    self.__post_init__()

            return new_init

        cls.__init__ = init_decorator(cls.__init__)


class RecPairDataset(RecNegativeDataset):
    """
    Dataset for pairwise loss function, generate pairwise data, return with userid, positive and negative itemid
    """

    def __init__(self, data, configs: DatasetConfig = None, **kwargs) -> None:
        super(RecPairDataset, self).__init__(data, configs=configs, **kwargs)
        self.concact = False
        self.label = False
        self.train_data = kwargs['total_data']['train'].copy().to_numpy()

    def __getitem__(self, index):
        return {
            'user': self.data[index][0],
            'positive': self.data[index][1],
            'negative': self.data[index][2],
            'positive_ips': self.ips_score.positive[self.data[index][1]],
            'negative_ips': self.ips_score.negative[self.data[index][2]],
            'negative_ips_dpr': self.ips_score.positive[self.data[index][2]],
        }


class RecPairExplainDataset(RecNegativeDataset):
    """
    Dataset for pairwise loss function, generate pairwise data, return with userid, positive and negative itemid
    """

    def __init__(self, data, configs: DatasetConfig = None, **kwargs) -> None:
        super(RecPairExplainDataset, self).__init__(data, configs=configs, **kwargs)
        self.concact = False
        self.label = False
        self.train_data = kwargs['total_data']['train'].copy().to_numpy()
        self.explain_score = self.cal_explain_matrix()

    def __getitem__(self, index):
        return {
            'user': self.data[index][0],
            'positive': self.data[index][1],
            'negative': self.data[index][2],
            'positive_ips': self.ips_score.positive[self.data[index][1]],
            'negative_ips': self.ips_score.negative[self.data[index][2]],
            'negative_ips_dpr': self.ips_score.positive[self.data[index][2]],
            'positive_explain': self.explain_score[self.data[index][0]][
                self.data[index][1]
            ],
            'negative_explain': self.explain_score[self.data[index][0]][
                self.data[index][2]
            ],
        }


class RecVaeDataset(RecDataset):
    def __init__(self, data, configs: DatasetConfig = None, **kwargs):
        super(RecVaeDataset, self).__init__(
            data, train_mask=False, configs=configs, **kwargs
        )
        self.users = np.unique(self.data[:, 0])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return {'user': self.users[index]}

    def histiory_items(self, value_field=None, max_history_len=None, row='user'):
        '''
        get the history items for each user
        @return: coo matrix,row: user , col: item
        '''
        self.data = pd.DataFrame(self.data.copy(), columns=['user', 'item', 'rating'])
        inter_feat = self.data.loc[self.data.rating > 0]
        # inter_feat.shuffle()
        user_ids, item_ids = (
            inter_feat['user'].to_numpy().astype(np.int32),
            inter_feat['item'].to_numpy().astype(np.int32),
        )
        if value_field is None:
            values = np.ones(len(inter_feat))
        else:
            if value_field not in inter_feat:
                raise ValueError(
                    f"Value_field [{value_field}] should be one of `inter_feat`'s features."
                )
            values = inter_feat[value_field].numpy()

        if row == "user":
            row_num, max_col_num = self.dataInfo['user'][0], self.dataInfo['item'][0]
            row_ids, col_ids = user_ids, item_ids
        else:
            row_num, max_col_num = self.dataInfo['item'][0], self.dataInfo['user'][0]
            row_ids, col_ids = item_ids, user_ids

        history_len = np.zeros(row_num, dtype=np.int64)
        for row_id in row_ids:
            history_len[int(row_id)] += 1

        max_inter_num = np.max(history_len)
        if max_history_len is not None:
            col_num = min(max_history_len, max_inter_num)
        else:
            col_num = max_inter_num

        history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        history_value = np.zeros((row_num, col_num))
        history_len[:] = 0
        for row_id, value, col_id in tqdm(
            iterable=zip(row_ids, values, col_ids),
            desc=f'Processing vae datasets:',
            ncols=80,
            colour='blue',
            total=len(values),
            bar_format='{desc}{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}{postfix}]',
        ):
            if history_len[row_id] >= col_num:
                continue
            history_matrix[row_id, history_len[row_id]] = col_id
            history_value[row_id, history_len[row_id]] = value
            history_len[row_id] += 1

        return {'matrix': history_matrix, 'value': history_value, 'len': history_len}


class RecSampleForEvalDataset(RecNegativeDataset):
    def __init__(self, data, configs: DatasetConfig = None, **kwargs):
        super(RecSampleForEvalDataset, self).__init__(data, configs=configs, **kwargs)
        # sample 99 negatives for each user-item pairs in test set
        self.negative_num = 99
        self.dataInfo = kwargs['dataInfo']

    def __post_init__(self):
        self.__sample__()
        print(
            Fore.GREEN
            + f'After sampling {self.negative_num} negative samples for sample-test or eval set, the dataset size is {self.__len__()}'
        )
        self.item_popularity = self.__calculate_item_popularity__(
            self.data[:, :2], self.dataInfo
        )

    def __getitem__(self, index):
        return {
            'user': self.data[index][0],
            'item': self.data[index][1],
            'item_pop': self.item_popularity[self.data[index][1]],
        }


class RecFullSortForTestDataset(RecDataset):
    def __init__(self, data, configs: DatasetConfig = None, **kwargs):
        # 对没一个用户给予所有物品进行计算，设置额外label来标识是否位于测试集和训练集中的正样本中，以便于后续metric的计算
        super(RecFullSortForTestDataset, self).__init__(
            data, configs=configs, binary=False, train_mask=False, **kwargs
        )
        if kwargs['total_data'] is None:
            raise ValueError(
                Fore.RED
                + 'Fullsort data need full data to generate label dataset, can not be None, please provide train and test data'
            )
        self.dataInfo = kwargs.pop('dataInfo', None)
        self.train_data = kwargs['total_data']['train']
        self.users = np.unique(
            self.data[self.data[:, 2] * 10 >= 8 * self.data[:, 2].max()][:, 0]
        )
        self.ground_truth, self.mask = self.__generate_ground_truth()
        self.item_popularity = self.__calculate_item_popularity__(
            self.train_data.copy().to_numpy()
        )

    def __len__(self):
        return len(self.users)

    def __generate_ground_truth(self):
        mask = pd.DataFrame(self.train_data, columns=['user', 'item', 'rating'])
        # mask = mask[mask.rating * 10 >= 8 * mask.rating.max()]
        mask = mask.groupby('user').item.apply(list)
        max_mask = np.max([len(a) for a in mask])
        mask = np.asarray(
            [np.pad(a, (0, max_mask - len(a)), 'symmetric') for a in mask]
        )

        ground_truth = pd.DataFrame(self.data, columns=['user', 'item', 'rating'])
        ground_truth = ground_truth[
            ground_truth.rating * 10 >= 8 * ground_truth.rating.max()
        ]
        ground_truth = ground_truth.groupby('user').item.apply(list)
        max_truth = np.max([len(a) for a in ground_truth])
        ground_truth = np.asarray(
            [np.pad(a, (0, max_truth - len(a)), 'symmetric') for a in ground_truth]
        )

        return ground_truth, mask

    def __getitem__(self, index):
        return {'user': self.users[index], 'id': index}
