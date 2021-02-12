from typing import Union

import numpy as np


def get_labels_set(labels: Union[tuple, list, np.ndarray], labels_per_task: int, shuffle_labels: bool = False,
                   random_state: Union[np.random.RandomState, int] = None):
    if shuffle_labels:
        if random_state is not None:
            if isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
            random_state.shuffle(labels)
        else:
            np.random.shuffle(labels)

    labels_sets = [list(labels[i:i + labels_per_task]) for i in range(0, len(labels), labels_per_task)]

    if len(labels_sets[-1]) == 1:
        labels_sets[-2].extend(labels_sets[-1])
        labels_sets = labels_sets[:-1]

    return np.asarray(labels_sets)