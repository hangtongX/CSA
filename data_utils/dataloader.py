from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
