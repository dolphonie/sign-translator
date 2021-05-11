# Created by Patrick Kao
from torch.utils.data import Dataset


class MaxLenWrapper(Dataset):
    """
    Removes elements greater than max len from the dataset
    """

    def __init__(self, dataset: Dataset, max_len: int):
        self.dataset = dataset
        self.max_len = max_len
        print(f"Filtering data points longer than {max_len}")
        self.filtered_indices = self._get_filtered_indices()
        print(f"Filtered {self.__len__() / len(self.dataset)} proportion of points")

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        return self.dataset[self.filtered_indices[idx]]

    def _get_filtered_indices(self):
        filtered_indices = []
        for i, item in enumerate(self.dataset):
            size = item["frames"].shape[0]
            if size <= self.max_len:
                filtered_indices.append(i)
        return filtered_indices


class MaxLenWrapperIterable(Dataset):
    def __init__(self, dataset: Dataset, max_len: int):
        self.dataset = dataset
        self.max_len = max_len
        self.include_count = 0
        self.exclude_indices = set()

    def __iter__(self):
        for i in range(len(self.dataset)):
            if i in self.exclude_indices:
                continue
            else:
                item = self.dataset[i]
                size = item["frames"].shape[0]
                if size <= self.max_len:
                    self.include_count += 1
                    yield item
                else:
                    self.exclude_indices.add(i)

    def get_included_excluded_counts(self):
        return self.include_count, len(self.exclude_indices)
