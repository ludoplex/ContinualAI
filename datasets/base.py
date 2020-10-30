from abc import ABC, abstractmethod
from os import makedirs
from os.path import join, dirname, exists
from typing import Callable, Union, Tuple

import numpy as np
from datasets.utils import split_dataset


class UnsupervisedDataset(object):
    """
    This class contains all the functions to operate with an unsupervised dataset
    (a dataset which does not contains labels).
    It allows to use transformation (pytorch style) and to have all the dataset split (train, test, split) in one place.
    it also possible to split the dataset using custom percentages of the whole dataset.
    :param x: The samples of the dataset.
    :param train: The list of the training set indexes.
    :param test: The list of the testing set indexes, if present.
    :param dev: The list of the dev set indexes, if present.
    :param transformer: A callable function f(x), which takes as input a sample and transforms it.
    In the case it is undefined, the identity function is used,
    :param kwargs: Additional parameters.
    """
    def __init__(self, x, train: list, test: list = None, dev: list = None, transformer: Callable = None, **kwargs):
        """
        The init function, used to instantiate the class.
        :param x: The samples of the dataset.
        :param train: The list of the training set indexes.
        :param test: The list of the testing set indexes, if present.
        :param dev: The list of the dev set indexes, if present.
        :param transformer: A callable function f(x), which takes as input a sample and transforms it.
        In the case it is undefined, the identity function is used,
        :param kwargs: Additional parameters.
        """
        super().__init__(**kwargs)

        if dev is None:
            dev = []
        if test is None:
            test = []

        assert len(x) == sum(map(len, [train, test, dev]))

        self._x = x
        self._train_split, self._test_split, self._dev_split = train, test, dev

        self._split = 'train'
        self._current_split_idx = self._train_split

        self.transformer = transformer if transformer is not None else lambda z: z

    def __getitem__(self, item) -> Tuple[Union[slice, tuple, int, list], np.ndarray]:
        """
        Given an index, or a list of indexes (defined as list, tuple o slice), return the associated samples.
        :param item: The index, or more than one, used to fetch the samples in the dataset.
        :return: Return :param item: and the associated samples, modified by the transformer function.
        """
        return item, self.transformer(self._x[item])

    def __len__(self):
        return len(self._current_split_idx)

    @property
    def split(self):
        """
        Return the current split of the dataset.
        :return: The return can be train, test or dev.
        """
        return self._split

    @property
    def x(self) -> np.ndarray:
        """
        Return the samples in the dataset.
        """
        return self.transformer(self._x[self._current_split_idx])

    @property
    def data(self) -> np.ndarray:
        """
        Return the samples in the dataset.
        """
        return self.x

    @property
    def train_split(self) -> list:
        """
        Return the train indexes,
        """
        return self._train_split

    @property
    def test_split(self) -> list:
        """
        Return the test indexes.
        """
        return self._test_split

    @property
    def dev_split(self) -> list:
        """
        Return the dev indexes.
        """
        return self._dev_split

    def train(self) -> None:
        """
        Set the current split to train.
        """
        self._split = 'train'
        self._current_split_idx = self.train_split

    def test(self) -> None:
        """
        Set the current split to test.
        """
        self._split = 'test'
        self._current_split_idx = self.test_split

    def dev(self) -> None:
        """
        Set the current split to dev.
        """
        self._split = 'dev'
        self._current_split_idx = self.dev_split

    def all(self) -> None:
        """
        Set the current split to the whole dataset.
        """
        self._split = 'all'
        self._current_split_idx = self.train_split + self.test_split + self.dev_split

    def preprocess(self, f: Callable) -> None:
        """
        Apply the input function f to the whole dataset.
        :param f: The callable function: f(x).
        """
        self._x = f(self._x)

    def split_dataset(self, test_split: float = 0.2, dev_split: float = 0.0,
                      random_state: Union[np.random.RandomState, int] = None) -> None:
        """
        When called, split the dataset according to the values passed to the function.
        Modify the dataset in-place.
        :param test_split: The percentage of the data to be used in the test set. Must be 0 <= test_split <= 1
        :param dev_split: The percentage of the data to be used in the dev set. Must be 0 <= dev_split <= 1.
        :param random_state: The random state used to shuffle the dataset.
        If it isn't a numpy RandomState object, the object is retrieved by doing
        np.random.RandomState(:param random_state:).
        """
        assert test_split >= 0, 'The test_split must be 0 <= test_split <= 1. The current value is {}'\
            .format(test_split)
        assert dev_split >= 0, 'The dev_split must be 0 <= dev_split <= 1. The current value is {}'.format(dev_split)
        assert test_split + dev_split < 1, 'The sum of test_split and dev_split must be less than 1. ' \
                                           'The current value is {}'.format(dev_split + test_split)

        self._train_split, self._test_split, self._dev_split = \
            split_dataset(self._x, test_split=test_split, dev_split=dev_split, random_state=random_state)


class SupervisedDataset(UnsupervisedDataset):
    def __init__(self, x, y, train, test=None, dev=None, transformer: Callable = None,
                 target_transformer: Callable = None, **kwargs):
        super().__init__(x, train, test, dev, transformer, **kwargs)

        self._y = y
        assert len(self._x) == len(self._y)

        self.target_transformer = target_transformer if target_transformer is not None else lambda z: z

        self._labels = sorted(list(set(y)))

    def __getitem__(self, item):
        item, x = super().__getitem__(item)
        y = self.target_transformer(self._y[item])

        return item, x, y

    @property
    def labels(self):
        return self.labels

    @property
    def y(self):
        return self.target_transformer(self._y[self._current_split_idx])

    @property
    def data(self):
        return self.x, self.y

    def preprocess_targets(self, f: Callable):
        self._y = f(self._y)

    def split_dataset(self, test_split: float = 0.2, dev_split: float = 0.0, balanced_split: bool = True,
                      random_state: Union[np.random.RandomState, int] = None):
        assert test_split >= 0
        assert dev_split >= 0
        assert test_split + dev_split < 1

        self._train_split, self._test_split, self._dev_split = \
            split_dataset(self._y, balance_labels=True,
                          test_split=test_split, dev_split=dev_split, random_state=random_state)


class DownloadableDataset(ABC):

    def __init__(self, name, transformer: Callable = None,
                 download_if_missing: bool = True, data_folder: str = None, **kwargs):

        if data_folder is None:
            data_folder = join(dirname(__file__), 'downloaded_datasets', name)

        self.data_folder = data_folder
        self._name = name

        self.transformer = transformer if transformer is not None else lambda x: x

        missing = not self._check_exists()

        if missing:
            if not download_if_missing:
                raise IOError("Data not found and `download_if_missing` is False")
            else:
                if not exists(self.data_folder):
                    makedirs(self.data_folder)

                print('Downloading dataset')
                self.download_dataset()

        # if not exists(self._data_folder):
        #     if not download_if_missing:
        #         if not self._check_exists():
        #             raise IOError("Data not found and `download_if_missing` is False")
        #     else:
        #         if not exists(self._data_folder):
        #             makedirs(self._data_folder)
        #
        #         print('Downloading dataset')
        #         self.download_dataset()

    @property
    def name(self):
        return self._name

    @abstractmethod
    def download_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def _check_exists(self) -> bool:
        raise NotImplementedError


class UnsupervisedDownloadableDataset(DownloadableDataset, UnsupervisedDataset, ABC):
    def __init__(self, name, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, target_transformer: Callable = None, **kwargs):
        super().__init__(name=name, transformer=transformer,
                         download_if_missing=download_if_missing, data_folder=data_folder)

        x, (train, test, dev) = self.load_dataset()

        super(DownloadableDataset, self).__init__(x=x, train=train, test=test, dev=dev,
                                                  transformer=transformer, target_transformer=target_transformer,
                                                  **kwargs)

    @abstractmethod
    def load_dataset(self) -> Tuple[np.ndarray, Tuple[list, list, list]]:
        raise NotImplementedError


class SupervisedDownloadableDataset(DownloadableDataset, SupervisedDataset, ABC):
    def __init__(self, name, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, target_transformer: Callable = None, **kwargs):
        super().__init__(name=name, transformer=transformer,
                         download_if_missing=download_if_missing, data_folder=data_folder)

        (x, y), (train, test, dev) = self.load_dataset()

        super(DownloadableDataset, self).__init__(x=x, y=y, train=train, test=test, dev=dev,
                                                  transformer=transformer, target_transformer=target_transformer,
                                                  **kwargs)

    @abstractmethod
    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[list, list, list]]:
        raise NotImplementedError
