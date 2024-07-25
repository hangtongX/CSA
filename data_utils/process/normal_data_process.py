import os.path
import numpy as np
import pandas as pd
from data_utils.data_process import DataProcess
from utils.plots import plot_line


class Yelp(DataProcess):
    def __init__(self, data_name='yelp'):
        super(Yelp, self).__init__(data_name=data_name)
        self.columns = ["user", "item", "rating", "timestamp"]

    def loadData(self):
        data_path = os.path.join(self.path, "yelp", "raw_data", "yelp2018.inter")
        data = pd.read_csv(data_path, sep="\t", header=0, engine="python")[
            ["user_id:token", "item_id:token", "rating:float", "timestamp:float"]
        ]
        data.columns = self.columns
        return data


class Mind(DataProcess):
    def __init__(self, data_name='mind'):
        super(Mind, self).__init__(data_name=data_name)
        self.columns = ["user", "item", "rating", "timestamp"]

    def loadData(self):
        data_path = os.path.join(self.path, "mind", "raw_data", "mind_small_dev.inter")
        data = pd.read_csv(
            data_path,
            sep="\t",
            header=0,
            engine="python",
            names=["user", "item", "rating", "timestamp"],
        )
        return data


class TaFeng(DataProcess):
    def __init__(self, data_name='tafeng'):
        super(TaFeng, self).__init__(data_name=data_name)
        self.columns = ["user", "item", "rating", "timestamp", "price"]

    def loadData(self):
        data_path = os.path.join(self.path, "tafeng", "raw_data", "ta-feng.inter")
        data = pd.read_csv(
            data_path,
            sep="\t",
            header=0,
            engine="python",
            names=["timestamp", "user", "item", "price"],
        )  #
        data["rating"] = pd.Series(np.ones_like(data.timestamp.to_numpy()))
        return data

    def regroup(self, user, item, data):
        return pd.DataFrame(
            data={
                "user": user,
                "item": item,
                'rating': data['rating'],
                'timestamp': data['timestamp'],
                'price': data['price'],
            }
        )


class Epinions(TaFeng):
    def __init__(self, data_name='epinions'):
        super(Epinions, self).__init__(data_name=data_name)

    def loadData(self):
        data_path = os.path.join(self.path, "epinions", "raw_data", "epinions.inter")
        data = pd.read_csv(
            data_path,
            sep="\t",
            header=0,
            engine="python",
            names=["user", "item", "rating", "timestamp", "price"],
        )
        return data


class LastFm(DataProcess):
    def __init__(self, data_name='lastfm'):
        super(LastFm, self).__init__(data_name=data_name)
        self.columns = ["user", "item", "rating", "timestamp"]

    def loadData(self):
        data_path = os.path.join(
            self.path, "lastfm", "raw_data", "user_taggedartists-timestamps.dat"
        )
        data = pd.read_csv(
            data_path,
            sep="\t",
            header=0,
            engine="python",
            names=["user", "item", "rating", "timestamp"],
        )
        data['rating'] = 1
        data.drop_duplicates(inplace=True)
        return data


class Reasoner(DataProcess):
    def __init__(self, data_name='reasoner'):
        super(Reasoner, self).__init__(data_name=data_name)
        self.columns = ["user", "item", "rating", "like"]

    def loadData(self):
        data_path = os.path.join(self.path, "reasoner", "raw_data", "interaction.csv")
        data = pd.read_csv(data_path, sep="\t", header=0, engine="python")[
            ["user_id", "video_id", "rating", "like"]
        ]
        data.columns = self.columns
        return data

    def regroup(self, user, item, data):
        return pd.DataFrame(
            data={
                "user": user,
                "item": item,
                'rating': data['rating'],
                'like': data['like'],
            }
        )

    def build(self):
        data = self.loadData()
        data = data[self.columns]
        print(
            f'Original data, there are {data.shape[0]} watching events from {data.user.unique().shape[0]} users '
            f'and {data.item.unique().shape[0]} items'
        )
        data.sort_values(by=["user"], inplace=True, ascending=[True])
        data.reset_index(inplace=True, drop=True)
        data = data[self.columns]
        data, user_count, item_count = self.filter_triplets(data)
        data = self.numerize(data)
        self.saveData(data[self.columns])
        print(data.head(5))


class Ifashion(Reasoner):
    def __init__(self, data_name='ifashion'):
        super(Ifashion, self).__init__(data_name=data_name)
        self.columns = ["user", "item", "rating"]

    def loadData(self):
        data_path = os.path.join(self.path, "ifashion", "raw_data", "outfit_data.txt")
        data = pd.read_csv(data_path, sep=",", header=None, engine="python")
        data.columns = ['user', 'items']
        data["item_list"] = data["items"].apply(lambda x: x.split(";"))
        arr = data.apply(
            lambda x: [[x["user"], item, 1] for item in x["item_list"]], axis=1
        ).tolist()
        data = pd.DataFrame(
            [a for b in arr for a in b], columns=["user", "item", "rating"]
        )
        data.sort_values(by=["user"], inplace=True, ascending=[True])
        data.reset_index(inplace=True, drop=True)
        data = data[["user", "item", "rating"]]
        return data

    def regroup(self, user, item, data):
        return pd.DataFrame(data={"user": user, "item": item, 'rating': data['rating']})


class Ml100K(DataProcess):
    def __init__(self, data_name='ml-100k'):
        super(Ml100K, self).__init__(data_name=data_name)
        self.columns = ["user", "item", "rating", "timestamp"]

    def loadData(self):
        data_path = os.path.join(self.path, "ml-100k", "raw_data", "u.data")
        data = pd.read_csv(data_path, sep="\t", header=0, engine="python")
        data.columns = self.columns
        return data


class Ml1M(DataProcess):
    def __init__(self, data_name='ml-1m'):
        super(Ml1M, self).__init__(data_name=data_name)
        self.columns = ["user", "item", "rating", "timestamp"]

    def loadData(self):
        data_path = os.path.join(self.path, self.data_name, "raw_data", "ratings.dat")
        data = pd.read_csv(data_path, sep="::", header=None, engine="python")
        data.columns = self.columns
        return data


class Ml10M(Ml1M):
    def __init__(self, data_name='ml-10m'):
        super(Ml10M, self).__init__(data_name=data_name)


class Ml20M(DataProcess):
    def __init__(self, data_name='ml-20m'):
        super(Ml20M, self).__init__(data_name=data_name)
        self.columns = ["user", "item", "rating", "timestamp"]

    def loadData(self):
        data_path = os.path.join(self.path, self.data_name, "raw_data", "ratings.csv")
        data = pd.read_csv(data_path, sep=",", header=None, engine="python")
        data.columns = self.columns
        return data


class Ml25M(Ml20M):
    def __init__(self, data_name='ml-25m'):
        super(Ml25M, self).__init__(data_name=data_name)


class Yahoo(DataProcess):
    def __init__(self, data_name='yahoo'):
        super(Yahoo, self).__init__(data_name=data_name)
        self.columns = ["user", "item", "rating"]

    def loadData(self):
        data_path = os.path.join(self.path, self.data_name, "raw_data")
        train_data = pd.read_csv(
            str(
                os.path.join(str(data_path), "ydata-ymusic-rating-study-v1_0-train.txt")
            ),
            sep='\t',
            header=None,
            engine="python",
        )
        test_data = pd.read_csv(
            str(data_path + '/ydata-ymusic-rating-study-v1_0-test.txt'),
            sep="\t",
            header=None,
            engine="python",
        )
        train_data.columns = self.columns
        test_data.columns = self.columns
        return train_data, test_data

    def build(self):
        train_data, test_data = self.loadData()
        train_data = train_data[self.columns]
        test_data = test_data[self.columns]
        print(len(train_data.user.unique()))
        print(
            f'Original data, there are {train_data.shape[0] + test_data.shape[0]} watching events '
            f'from {train_data.user.unique().shape[0]} users '
            f'and {train_data.item.unique().shape[0]} items'
        )
        train_data.sort_values(by=["user"], inplace=True, ascending=[True])
        train_data.reset_index(inplace=True, drop=True)
        test_data.sort_values(by=["user"], inplace=True, ascending=[True])
        test_data.reset_index(inplace=True, drop=True)
        train_path = os.path.join(self.path, self.data_name, 'train.pkl')
        test_path = os.path.join(self.path, self.data_name, 'test.pkl')
        train_data.to_pickle(train_path, compression=self.compression)
        test_data.to_pickle(test_path, compression=self.compression)


class Coat(DataProcess):
    def __init__(self, data_name='coat'):
        super(Coat, self).__init__(data_name=data_name)

    def loadData(self, name='train'):
        data_path = os.path.join(self.path, self.data_name, "raw_data/")
        data = np.loadtxt(str(data_path) + name + '.ascii')
        row, col = np.nonzero(data)
        label = np.ones_like(row).reshape(-1, 1)
        data = np.hstack((row.reshape(-1, 1), col.reshape(-1, 1)))
        data = np.hstack((data, label))
        data = pd.DataFrame(data, columns=['user', 'item', 'rating'])
        return data

    def build(self):
        train_data = self.loadData(name='train')
        test_data = self.loadData(name='test')
        print(
            f'Original data, there are {train_data.shape[0] + test_data.shape[0]} watching events '
            f'from {train_data.user.unique().shape[0]} users '
            f'and {train_data.item.unique().shape[0]} items'
        )
        train_path = os.path.join(self.path, self.data_name, 'train.pkl')
        test_path = os.path.join(self.path, self.data_name, 'test.pkl')
        train_data.to_pickle(str(train_path), compression=self.compression)
        test_data.to_pickle(str(test_path), compression=self.compression)

    def plot(self):
        train = pd.read_pickle(self.path + '/coat/train.pkl', compression='gzip')
        test = pd.read_pickle(self.path + '/coat/test.pkl', compression='gzip')
        data = np.hstack(
            (
                train.groupby('item').user.count().to_numpy().reshape(-1, 1),
                test.groupby('item').user.count().to_numpy().reshape(-1, 1),
            )
        )
        data = data[np.argsort(-data[:, 0])]
        plot_line(
            data_x=range(data.shape[0]),
            data_y=data,
            x_label='Items sorted by counts',
            y_label='Interaction counts',
            path=str(os.path.join(self.path, self.data_name, 'plot.png')),
            labels=['Partially Observed (MNAR)', 'Full Observed (MAR)'],
            is_fit=True,
        )


class Kuairec(DataProcess):
    def __init__(self, data_name='kuairec'):
        super(Kuairec, self).__init__(data_name=data_name)
        self.columns = ["user", "item", "rating"]

    def loadData(self):
        data_path = str(os.path.join(self.path, self.data_name, "raw_data/"))
        train = pd.read_csv(data_path + 'big_matrix.csv')[
            ['user_id', 'video_id', 'watch_ratio']
        ]
        train['watch_ratio'].iloc[train['watch_ratio'] < 2.0] = 0
        train['watch_ratio'].iloc[train['watch_ratio'] >= 2.0] = 1
        test = pd.read_csv(data_path + 'small_matrix.csv')[
            ['user_id', 'video_id', 'watch_ratio']
        ]
        test['watch_ratio'].iloc[test['watch_ratio'] < 2.0] = 0
        test['watch_ratio'].iloc[test['watch_ratio'] >= 2.0] = 1
        test.columns = self.columns
        train.columns = self.columns

        return train, test

    def build(self):
        train_data, test_data = self.loadData()
        print(
            f'Original data, there are {train_data.shape[0] + test_data.shape[0]} watching events '
            f'from {train_data.user.unique().shape[0]} users '
            f'and {train_data.item.unique().shape[0]} items'
        )
        train_path = os.path.join(self.path, self.data_name, 'train.pkl')
        test_path = os.path.join(self.path, self.data_name, 'test.pkl')
        train_data.to_pickle(str(train_path), compression=self.compression)
        test_data.to_pickle(str(test_path), compression=self.compression)

    def plot(self):
        train, test = self.loadData()
        items = test.item.unique()
        train = train[train['item'].isin(items)]
        data = np.hstack(
            (
                train.groupby('item').rating.sum().to_numpy().reshape(-1, 1),
                test.groupby('item').rating.sum().to_numpy().reshape(-1, 1),
            )
        )
        data = data[np.argsort(-data[:, 0])]
        plot_line(
            data_x=range(data.shape[0]),
            data_y=data,
            x_label='Items sorted by counts',
            y_label='Interaction counts',
            path=str(os.path.join(self.path, self.data_name, 'plot.png')),
            labels=['Partially Observed (MNAR)', 'Full Observed (MNAR)'],
            is_fit=True,
        )


Reasoner().build()
