import os
import pickle
from typing import Literal, Tuple, Any
from collections import OrderedDict
import numpy as np
import pandas as pd


class DataProcess(object):
    def __init__(
        self,
        data_path=None,
        data_name=None,
        data_type='rec',
        compression: Literal['gzip'] = 'gzip',
    ):
        self.path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'normal_data',
        )
        self.columns = None
        self.data_name = data_name
        self.data_type = data_type
        self.compression = compression

    def process_data(self):
        process_methods = {'dsprites': None}
        back = process_methods[self.data_name]()
        if self.data_type == 'rec':
            static = self.save_rec_static(back['data'])
            return {**back, 'static': static}
        else:
            return back

    @staticmethod
    def get_count(data, id):
        return data[[id]].groupby(id, as_index=False).size()

    def filter_triplets(self, data, min_uc=0, min_sc=10):
        # Only keep the triplets for items which were clicked on by at least min_sc users.
        if min_sc > 0:
            itemcount = self.get_count(data, "item")
            data = data[
                data["item"].isin(itemcount['item'][itemcount['size'] >= min_sc])
            ]

        # Only keep the triplets for users who clicked on at least min_uc items
        # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
        if min_uc > 0:
            usercount = self.get_count(data, "user")
            data = data[
                data["user"].isin(usercount["user"][usercount["size"] >= min_uc])
            ]

        # Update both usercount and itemcount after filtering
        user_count, item_count = (
            self.get_count(data, "user").shape[0],
            self.get_count(data, "item").shape[0],
        )
        self.data_static(data, user_count, item_count)

        return data, user_count, item_count

    def data_static(self, data, user_count, item_count):
        sparsity = 1.0 * data.shape[0] / (user_count * item_count)

        print(
            "After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)"
            % (data.shape[0], user_count, item_count, sparsity * 100)
        )

    def numerize(self, data):
        # creat id mapping
        unique_user = data.user.unique()
        unique_item = data.item.unique()
        user_map = dict((uid, i) for i, uid in enumerate(unique_user))
        item_map = dict((iid, i) for i, iid in enumerate(unique_item))

        # save mapping dict
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with open(
            os.path.join(self.path, self.data_name, "user_map.pkl"), mode="wb"
        ) as file:
            pickle.dump(user_map, file)
        with open(
            os.path.join(self.path, self.data_name, "item_map.pkl"), mode="wb"
        ) as file:
            pickle.dump(item_map, file)
        # transform
        user = map(lambda x: user_map[x], data["user"])
        item = map(lambda x: item_map[x], data["item"])
        return self.regroup(user, item, data)

    def regroup(self, user, item, data):
        return pd.DataFrame(
            data={
                "user": user,
                "item": item,
                'rating': data['rating'],
                'timestamp': data['timestamp'],
            }
        )

    def saveData(self, data, name="data.pkl"):
        path = os.path.join(self.path, self.data_name, name)
        print(f"saving data to {path}")
        data.to_pickle(path, compression=self.compression)

    def save_rec_static(self, data):
        static = {
            'name': self.data_name,
            'user': int(data.user.max()) + 1,
            'item': int(data.item.max()) + 1,
        }
        path = os.path.join(self.path, self.data_name, 'static.pkl')
        print(f"saving data static to {path}")
        pd.DataFrame([static]).to_pickle(path, compression=self.compression)
        return static

    def loadData(self) -> pd.DataFrame:
        pass

    def build(self):
        data = self.loadData()
        data = data[self.columns]
        print(
            f'Original data, there are {data.shape[0]} watching events from {data.user.unique().shape[0]} users '
            f'and {data.item.unique().shape[0]} items'
        )
        data.sort_values(by=["user", "timestamp"], inplace=True, ascending=[True, True])
        data.reset_index(inplace=True, drop=True)
        data = data[self.columns]
        data, user_count, item_count = self.filter_triplets(data)
        data = self.numerize(data)
        self.saveData(data[self.columns])
        print(data.head(5))


class DatasetOutput(OrderedDict):
    """
    Base DatasetOutput class fixing the output type from the dataset. This class is inspired from
       the ``ModelOutput`` class from hugginface transformers library
    """

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any, ...]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())
