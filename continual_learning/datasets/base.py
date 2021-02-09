import os
from abc import ABC, abstractmethod
from os import makedirs
from os.path import join, dirname, exists
from typing import Callable, Union, Tuple, List

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from .utils import extract_dev, split_dataset


class IterableDataset:
    tensors: Tuple[Union[List, np.ndarray]]

    def __init__(self, *tensors: Union[list, np.ndarray]) -> None:
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])


class UnsupervisedDataset(object):
    # """
    # This class contains all the functions to operate with an unsupervised dataset
    # (a dataset which does not contains labels).
    # It allows to use transformation (pytorch style) and to have all the dataset split (train, test, split) in one place.
    # it also possible to split the dataset using custom percentages of the whole dataset.
    # :param x: The samples of the dataset.
    # :param train: The list of the training set indexes.
    # :param test: The list of the testing set indexes, if present.
    # :param dev: The list of the dev set indexes, if present.
    # :param transformer: A callable function f(x), which takes as input a sample and transforms it.
    # In the case it is undefined, the identity function is used,
    # :param kwargs: Additional parameters.
    # """

    def __init__(self, x, train: Union[list, np.ndarray], test: [list, np.ndarray] = None,
                 dev: [list, np.ndarray] = None,
                 transformer: Callable = None,
                 is_path_dataset: bool = False, images_path: str = '', **kwargs):

        """
        The init function, used to instantiate the class.
        :param x: The samples of the dataset.
        :param train: The list of the training set indexes.
        :param test: The list of the testing set indexes, if present.
        :param dev: The list of the dev set indexes, if present.
        :param transformer: A callable function f(x), which takes as input a sample and transforms it. In the case it is undefined, the identity function is used,
        :param is_path_dataset: If the dataset contains paths instead of images.
        :param images_path: the path from the root of the dataset, in which the images are stored.
        :param kwargs: Additional parameters.
        """

        super().__init__(**kwargs)

        if dev is None:
            dev = []
        if test is None:
            test = []

        assert len(x) == sum(map(len, [train, test, dev]))

        self.is_path_dataset = is_path_dataset
        self.images_path = images_path

        self._x = x
        self._train_split, self._test_split, self._dev_split = np.asarray(train, dtype=int), \
                                                               np.asarray(test, dtype=int), \
                                                               np.asarray(dev, dtype=int)

        self._split = 'train'
        self._current_split_idx = self._train_split

        self._transformer = transformer if transformer is not None else lambda z: z
        self.transformer = transformer if transformer is not None else lambda z: z

    def apply_transformer(self, v: bool = True):
        if v:
            self._transformer = self.transformer
        else:
            self._transformer = lambda x: x

    def __getitem__(self, item: Union[slice, int, list, np.ndarray]) -> Tuple[
        Union[list, tuple, int, list], np.ndarray]:
        """
        Given an index, or a list of indexes (defined as list, tuple o slice), return the associated samples.
        :param item: The index, or more than one, used to fetch the samples in the dataset.
        :return: Return :param item: and the associated samples, modified by the transformer function.
        """
        if self.is_path_dataset:
            img = Image.open(os.path.join(self.images_path, self._x[item]))
            img = img.convert('RGB')
            img = np.asarray(img)
            return item, self._transformer(img)
        else:
            if isinstance(item, slice):
                s = item.start if item.start is not None else 0
                e = item.stop
                step = item.step if item.step is not None else 1
                i = list(range(s, e, step))
            else:
                i = item

            # else:
            #     print(item)
            #     s = item.start if item.start is not None else 0
            #     e = item.stop
            #     step = item.step if item.step is not None else 1
            #     i = list(range(s, e, step))
            # print(type(self.current_indices))
            # a = self.current_indices[item]
            return i, self._transformer(self._x[self.current_indices[item]])

    def __len__(self):
        return len(self._current_split_idx)

    def get_iterable(self, batch_size, shuffle=True, sampler=None):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

    @property
    def current_split(self):
        """
        Return the current split of the dataset.
        :return: The return can be train, test or dev.
        """
        return self._split

    # @property
    # def split_indexes(self):
    #     return self._current_split_idx

    @property
    def current_indices(self):
        return self._current_split_idx

    @property
    def x(self) -> np.ndarray:
        """
        Return the samples of the current split.
        """
        return self._transformer(self._x[self.current_indices])

    @property
    def data(self) -> np.ndarray:
        """
        Alias for self.x.
        """
        return self.x

    @property
    def train_indices(self) -> np.ndarray:
        """
        Return the train indexes.
        """
        return self._train_split

    @property
    def test_indices(self) -> np.ndarray:
        """
        Return the test indexes.
        """
        return self._test_split

    @property
    def dev_indices(self) -> np.ndarray:
        """
        Return the dev indexes.
        """
        return self._dev_split

    def _get_subset(self, idx):
        x = [self._transformer(self._x[i]) for i in idx]
        return idx, x

    @property
    def train_set(self):
        return IterableDataset(self.train_indices, self._get_subset(self.train_indices))

    @property
    def dev_set(self):
        return IterableDataset(self.dev_indices, self._get_subset(self.dev_indices))

    @property
    def test_set(self):
        return IterableDataset(self.test_indices, self._get_subset(self.test_indices))

    def train(self) -> None:
        """
        Set the current split to train.
        """
        self._split = 'train'
        self._current_split_idx = self.train_indices

    def test(self) -> None:
        """
        Set the current split to test.
        """
        self._split = 'test'
        self._current_split_idx = self.test_indices

    def dev(self) -> None:
        """
        Set the current split to dev.
        """
        self._split = 'dev'
        self._current_split_idx = self.dev_indices

    def all(self) -> None:
        """
        Set the current split to the whole dataset.
        """
        self._split = 'all'
        self._current_split_idx = np.concatenate((self.train_indices, self.test_indices, self.dev_indices))

    def preprocess(self, f: Callable) -> None:
        """
        Apply the input function f to the current split.
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
        assert test_split >= 0, 'The test_split must be 0 <= test_split <= 1. The current value is {}' \
            .format(test_split)
        assert dev_split >= 0, 'The dev_split must be 0 <= dev_split <= 1. The current value is {}'.format(dev_split)
        assert test_split + dev_split < 1, 'The sum of test_split and dev_split must be less than 1. ' \
                                           'The current value is {}'.format(dev_split + test_split)

        self._train_split, self._test_split, self._dev_split = \
            split_dataset(self._x, test_split=test_split, dev_split=dev_split, random_state=random_state)


class SupervisedDataset(UnsupervisedDataset):
    """
    This class contains all the functions to operate with an supervised dataset.
    It allows to use transformation (pytorch style) and to have all the dataset split (train, test, split) in one place.
    it also possible to split the dataset using custom percentages of the whole dataset.
    :param x: The samples of the dataset.
    :param y: The labels associated to x
    :param train: The list of the training set indexes.
    :param test: The list of the testing set indexes, if present.
    :param dev: The list of the dev set indexes, if present.
    :param transformer: A callable function f(x), which takes as input a sample and transforms it.
    In the case it is undefined, the identity function is used,
    :param kwargs: Additional parameters.
    """

    def __init__(self, x, y, train, test=None, dev=None, transformer: Callable = None,
                 target_transformer: Callable = None, **kwargs):
        """
        :param x: The samples of the dataset.
        :param y: The labels associated to x
        :param train: The list of the training set indexes.
        :param test: The list of the testing set indexes, if present.
        :param dev: The list of the dev set indexes, if present.
        :param transformer: A callable function f(x), which takes as input a sample and transforms it.
        In the case it is undefined, the identity function is used,
        :param kwargs: Additional parameters.
        """
        super().__init__(x, train, test, dev, transformer, **kwargs)

        self._y = y
        assert len(self._x) == len(self._y)

        self.target_transformer = target_transformer if target_transformer is not None else lambda z: z
        self._target_transformer = target_transformer if target_transformer is not None else lambda z: z

        self._labels = tuple(sorted(list(set(y))))

    def __getitem__(self, item) -> Tuple[Union[list, tuple, int, list], np.ndarray, np.ndarray]:
        """
        Given an index, or a list of indexes (defined as list, tuple o slice), return the associated samples.
        :param item: The index, or more than one, used to fetch the samples in the dataset.
        :return: Return :param item: and the associated samples, x and y, modified by the transformer function.
        """
        i, x = super().__getitem__(item)
        return i, x, self._target_transformer(self._y[self.current_indices[item]])

    @property
    def labels(self) -> tuple:
        """
        :return: The set of labels of the dataset.
        """
        return self._labels

    def _get_subset(self, idx):
        _, x = super()._get_subset(idx)
        y = [self._target_transformer(self._y[i]) for i in idx]

        return x, y

    @property
    def train_set(self):
        x, y = self._get_subset(self.train_indices)
        return IterableDataset(self.train_indices, x, y)

    @property
    def dev_set(self):
        x, y = self._get_subset(self.dev_indices)
        return IterableDataset(self.dev_indices, x, y)

    @property
    def test_set(self):
        x, y = self._get_subset(self.test_indices)
        return IterableDataset(self.test_indices, x, y)

    @property
    def y(self):
        """
        :return: The labels of the current split.
        """
        return self._target_transformer(self._y[self.current_indices])

    @property
    def data(self):
        """
        Alias for (self.x, self.y)
        :return:
        """
        return self.x, self.y

    def apply_transformer(self, v: bool = True):
        super().apply_transformer(v)
        if v:
            self._target_transformer = self.target_transformer
        else:
            self._target_transformer = lambda x: x

    def preprocess_targets(self, f: Callable):
        """
        Apply the input function f to the current labels in the split.
        :param f: The callable function: f(x).
        """
        self._y = f(self._y)

    def split_dataset(self, test_split: float = 0.2, dev_split: float = 0.0, balanced_split: bool = True,
                      random_state: Union[np.random.RandomState, int] = None):
        """
        When called, split the dataset according to the values passed to the function.
        Modify the dataset in-place.
        :param test_split: The percentage of the data to be used in the test set. Must be 0 <= test_split <= 1
        :param dev_split: The percentage of the data to be used in the dev set. Must be 0 <= dev_split <= 1.
        :param balanced_split: If the resulting splits need to have balanced number of samples for each label
        :param random_state: The random state used to shuffle the dataset.
        If it isn't a numpy RandomState object, the object is retrieved by doing
        np.random.RandomState(:param random_state:).
        """
        assert test_split >= 0
        assert dev_split >= 0
        assert test_split + dev_split < 1

        self._train_split, self._test_split, self._dev_split = \
            split_dataset(self._y, balance_labels=True,
                          test_split=test_split, dev_split=dev_split, random_state=random_state)

    def create_dev_split(self, dev_split: float = 0.1,
                         random_state: Union[np.random.RandomState, int] = None):
        """
        When called, extract the dev split from the training set
        Modify the dataset in-place.
        :param dev_split: The percentage of the data to be used in the dev set. Must be 0 <= dev_split <= 1.
        :param random_state: The random state used to shuffle the dataset.
        If it isn't a numpy RandomState object, the object is retrieved by doing
        np.random.RandomState(:param random_state:).
        """
        assert dev_split >= 0

        self._train_split, self._dev_split = extract_dev(y=self.train_indices, dev_split=dev_split,
                                                         random_state=random_state)


class DownloadableDataset(ABC):

    def __init__(self, name, transformer: Callable = None,
                 download_if_missing: bool = True, data_folder: str = None, **kwargs):
        """
        An abstract class used to download the datasets.
        :param name: The name of the dataset.
        :param transformer: The transformer function used when a sample is retrieved.
        :param download_if_missing: If the dataset needs to be downloaded if missing.
        :param data_folder: Where the dataset is stored.
        """

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
        """
        :return: The name of the dataset
        """
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
                         download_if_missing=download_if_missing, data_folder=data_folder, **kwargs)
        (x, y), (train, test, dev) = self.load_dataset()

        if kwargs.get('is_path_dataset', False):
            kwargs['images_path'] = os.path.join(self.data_folder, kwargs['images_path'])

        super(DownloadableDataset, self).__init__(x=x, y=y, train=train, test=test, dev=dev,
                                                  transformer=transformer, target_transformer=target_transformer,
                                                  **kwargs)

    @abstractmethod
    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[list, list, list]]:
        raise NotImplementedError
