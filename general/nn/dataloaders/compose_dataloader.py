import random
from functools import reduce


class ComposeIterator:

    def __init__(self, iterators, shuffle=False):
        """
        Initialize the iterators.

        Args:
            self: write your description
            iterators: write your description
            shuffle: write your description
        """

        self.iterators = iterators
        self.shuffle = shuffle
        self._length = sum([len(it) for it in iterators])


    def __iter__(self):
        """
        Iterate over the iterators.

        Args:
            self: write your description
        """
        self.current_iterators = [
            iter(it) for it in self.iterators
        ]
        return self

    def __next__(self):
        """
        Returns the next item from the list.

        Args:
            self: write your description
        """

        while len(self.current_iterators) != 0:
            if self.shuffle:
                idx = random.choice(range(len(self.current_iterators)))
            else:
                idx = 0
            iterator = self.current_iterators[idx]
            try:
                return next(iterator)
            except StopIteration:
                del self.current_iterators[idx]
        raise StopIteration

    def __len__(self):
        """
        Returns the length of the sequence.

        Args:
            self: write your description
        """
        return self._length

    def __getitem__(self, idx):
        """
        Return the iterator at the given index.

        Args:
            self: write your description
            idx: write your description
        """
        return self.iterators[idx]


class ComposeDataloaders(ComposeIterator):

    def __init__(self, iterators, shuffle=False):
        """
        Initialize the chain.

        Args:
            self: write your description
            iterators: write your description
            shuffle: write your description
        """
        super().__init__(iterators, shuffle)

        datasets = [it.dataset for it in iterators]
        # self.dataset = sum(datasets[1:], datasets[0])
        self.dataset = reduce(lambda a, b: a+b, datasets)

    @property
    def batch_size(self):
        """
        Batch size of the first iterator in the list.

        Args:
            self: write your description
        """
        return self.iterators[0].batch_size
