import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_utils.data_process import DataProcess
from data_utils.dataset_config import DatasetConfig
from colorama import Fore, init as colorinit

colorinit(autoreset=True)


class DataLoad(object):
    def __init__(self, config: DatasetConfig):
        self.testSize = config.testSize
        self.trainSize = config.trainSize
        self.valSize = config.valSize
        if self.trainSize + self.valSize + self.testSize != 1:
            raise ValueError(
                f"The input partition ratio must be 1, which is now {self.trainSize + self.valSize + self.testSize}"
            )
        self.dataname = config.dataname
        self.path = self.data_path()
        self.compression = config.compression
        self.split_type = config.split_type
        self.split_data = config.split
        self.percent = [self.trainSize, self.valSize, self.testSize]
        self.process = DataProcess(self.path, self.dataname, self.split_type)

    def load(self):
        if self.split_type in ['rec']:
            return self.load_rec()
        elif self.split_type in ['img']:
            return self.load_img()

    def load_img(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError('The Dataset is not exist. please check.......')
        if not os.path.exists(os.path.join(self.path, 'train.npy')) or self.split_data:
            if os.path.exists(os.path.join(self.path, 'img.npy')):
                data = np.load(os.path.join(self.path, 'img.npy'))
            else:
                data = self.process.process_data()
            train, test, val = self.split(data=data)
        else:
            train = np.load(os.path.join(self.path, 'train.npy'))
            test = np.load(os.path.join(self.path, 'test.npy'))
            val = np.load(os.path.join(self.path, 'val.npy'))

        return {'train': train, 'test': test, 'val': val}

    def load_rec(self):
        """
        load data from dataname.pkl, if file not exist, load from Raw data and process it and save for next.
        @return: data after spilt by 0.7,0.1,0.2 for train, val, and test in default.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError('The Dataset is not exist. please check.......')
        if not os.path.exists(os.path.join(self.path, 'train.pkl')) or self.split_data:
            if os.path.exists(os.path.join(self.path, 'data.pkl')):
                data = pd.read_pickle(
                    self.path + '/data.pkl', compression=self.compression
                )
            else:
                data = self.process.process_data()
            self.save_rec_static(data)
            train, test, val = self.split(data=data)
        else:
            train = pd.read_pickle(
                self.path + '/train.pkl', compression=self.compression
            )
            test = pd.read_pickle(self.path + '/test.pkl', compression=self.compression)
            try:
                val = pd.read_pickle(
                    self.path + '/val.pkl', compression=self.compression
                )
            except FileNotFoundError as e:
                print(
                    Fore.YELLOW
                    + f'Data {self.dataname} loss val.pkl file, begin to generate from train.pkl'
                )
                self.save_rec_static(train)
                _, val = train_test_split(
                    test, test_size=self.valSize / (self.testSize + self.valSize)
                )
                # self.save([train, val], name=['train', 'val'])

        static = pd.read_pickle(
            self.path + '/static.pkl', compression=self.compression
        ).to_dict(orient='list')

        return {'train': train, 'test': test, 'val': val, 'datainfo': static}

    def load_full_data(self):
        if not os.path.exists(os.path.join(self.path, 'train.pkl')):
            if os.path.exists(os.path.join(self.path, 'data.pkl')):
                data = pd.read_pickle(
                    self.path + '/data.pkl', compression=self.compression
                )
            else:
                data = self.process.process_data()
        else:
            try:
                data = pd.read_pickle(
                    self.path + '/data.pkl', compression=self.compression
                )
            except FileNotFoundError as e:
                print(
                    Fore.YELLOW
                    + f'Dataset {self.dataname} lose data.pkl file, next try to load train.pkl...'
                )
                data = pd.read_pickle(
                    self.path + '/train.pkl', compression=self.compression
                )

        return data

    def load_static(self):
        if not os.path.exists(self.path + 'static.pkl'):
            static = self.load()['datainfo']
        else:
            static = pd.read_pickle(
                self.path + 'static.pkl', compression=self.compression
            ).to_dict(orient='list')
        return static

    def split(self, data):
        train, test, val = None, None, None
        if self.split_type == 'rec':

            grouped_inter_feat_index = self.grouped_index(data['user'].to_numpy())
            next_index = [[] for _ in range(len(self.percent))]
            for grouped_index in grouped_inter_feat_index:
                tot_cnt = len(grouped_index)
                split_ids = self.calcu_split_ids(tot=tot_cnt, ratios=self.percent)
                for index, start, end in zip(
                    next_index, [0] + split_ids, split_ids + [tot_cnt]
                ):
                    index.extend(grouped_index[start:end])
            train, val, test = [data.iloc[index] for index in next_index]
            train = pd.DataFrame(train, columns=['user', 'item', 'rating'])
            test = pd.DataFrame(test, columns=['user', 'item', 'rating'])
            val = pd.DataFrame(val, columns=['user', 'item', 'rating'])

        elif self.split_type in {'normal', 'img'}:
            train, val_test = train_test_split(
                data, test_size=self.valSize + self.testSize
            )
            val, test = train_test_split(
                val_test, test_size=self.testSize / (self.testSize + self.valSize)
            )
        else:
            raise TypeError('Invalid split type..........')

        self.save([train, test, val], ['train', 'test', 'val'])
        return train, val, test

    def save(self, data: list, name: list):
        if self.split_type in {'normal', 'rec'}:
            if not isinstance(data[0], pd.DataFrame):
                raise TypeError(
                    '\033[1;35;40m The input data must be the list of dataframe..............\033[0m'
                )
            if len(data) != len(name):
                raise ValueError(
                    '\033[1;35;40m The size of name and data must match.......\033[0m'
                )
            for idx, df in enumerate(data):
                df.to_pickle(
                    self.path + f'/{name[idx]}.pkl', compression=self.compression
                )
            print(f'\033[0;35;40m Total save {len(name)} files, contains {name}\033[0m')
        elif self.split_type in ['img']:
            if not isinstance(data[0], np.ndarray):
                raise TypeError(
                    '\033[1;35;40m The input data must be the list of numpy ndarray..............\033[0m'
                )
            if len(data) != len(name):
                raise ValueError(
                    '\033[1;35;40m The size of name and data must match.......\033[0m'
                )
            for idx, df in enumerate(data):
                np.save(self.path + f'/{name[idx]}.npy', df)
            print(f'\033[0;35;40m Total save {len(name)} files, contains {name}\033[0m')
        else:
            raise KeyError('Invalid split.......')

    def data_path(self) -> str:
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        datapath = {
            'yahoo': 'data/normal_data/yahoo/',
            'coat': 'data/normal_data/coat/',
            'dsprites': 'data/simulated_data/dsprites/',
            'ml-100k': 'data/normal_data/ml-100k/',
            'ml-1m': 'data/normal_data/ml-1m/',
            'ml-10m': 'data/normal_data/ml-10m/',
            'ml-20m': 'data/normal_data/ml-20m/',
            'ml-25m': 'data/normal_data/ml-25m/',
            'reasoner': 'data/normal_data/reasoner/',
            'epinions': 'data/normal_data/epinions/',
            'ifashion': 'data/normal_data/ifashion/',
            'lastfm': 'data/normal_data/lastfm/',
            'mind': 'data/normal_data/mind/',
            'yelp': 'data/normal_data/yelp/',
            'amazon': 'data/normal_data/amazon/',
            'kuairec': 'data/normal_data/kuairec/',
            'tafeng': 'data/normal_data/tafeng/',
        }
        if self.dataname not in datapath:
            raise ValueError('dataset not available...........')

        return str(os.path.join(path, datapath[self.dataname]))

    @staticmethod
    def grouped_index(group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index.values()

    @staticmethod
    def calcu_split_ids(tot, ratios):
        """
        Given split ratios, and total number, calculate the number of each part after splitting.
        Other than the first one, each part is rounded down.
        @param tot: Total number.
        @param ratios:  List of split ratios. No need to be normalized.
        @return: Number of each part after splitting.
        """
        cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
        cnt[0] = tot - sum(cnt[1:])
        for i in range(1, len(ratios)):
            if cnt[0] <= 1:
                break
            if 0 < ratios[-i] * tot < 1:
                cnt[-i] += 1
                cnt[0] -= 1
        split_ids = np.cumsum(cnt)[:-1]
        return list(split_ids)

    def load_or_generate_negatives(self):
        """
        Load negatives from xxx/negative.pkl or generate if file doesn't exist
        :return: dataframe with colums [user, postives, negatives]
        :rtype: dataframe
        """
        try:
            if os.access(self.path + 'negative.npz', mode=os.F_OK):
                negative = pd.read_pickle(
                    self.path + 'negative.pkl', compression=self.compression
                )
                return negative
            else:
                raise FileExistsError(
                    Fore.YELLOW
                    + f'no exist negative file of {self.dataname}, begain to genreate.......'
                )
        except FileExistsError as e:
            print(e)
            data = self.load_full_data()
            item_pool = set(data.item.unique())
            interact_status = (
                data.groupby('user')['item']
                .apply(set)
                .reset_index()
                .rename(columns={'item': 'postive_items'})
                .sort_values(by='user', ascending=True)
            )
            interact_status['negative_items'] = ''
            interact_status['negative_items'] = interact_status[
                'negative_items'
            ].astype('object')
            for idx, item in tqdm(interact_status.iterrows()):
                interact_status.at[idx, 'negative_items'] = (
                    item_pool - interact_status.loc[idx, 'postive_items']
                )
            # interact_status['negative_items'] = (
            #     item_pool - interact_status['postive_items']
            # )
            # data = interact_status[
            #     ['user', 'postive_items', 'negative_items']
            # ].sort_values(by='user', ascending=True)
            print(Fore.GREEN + 'Negatives successfully generated')
        return data

    def load_item_pool(self):
        data = self.load_full_data()
        item_pool = set(data.item.unique())
        return item_pool

    def save_rec_static(self, data):
        static = {
            'name': self.dataname,
            'user': int(data.user.max()) + 1,
            'item': int(data.item.max()) + 1,
            'total': data.shape[0],
        }
        pd.DataFrame([static]).to_pickle(
            self.data_path() + 'static.pkl', compression=self.compression
        )
        return static
