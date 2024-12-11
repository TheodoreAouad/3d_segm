class ListIndex(list):

    def __getitem__(self, idx):
        """
        Return an item from the container.

        Args:
            self: write your description
            idx: write your description
        """
        if isinstance(idx, int):
            return super().__getitem__(idx)
        if not isinstance(idx, tuple):
            raise ValueError("index must be int or tuple.")

        itm = super().__getitem__(idx[0])
        for cur_idx in idx[1:]:
            itm = itm[cur_idx]

        return itm
