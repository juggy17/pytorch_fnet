import util.data

class TestImgDataProvider(object):
    """Wrapper class to provider larger chunks for model testing."""
    def __init__(self, dataset, dims_chunk=(32, 128, 128), dims_pin=(None, None, None)):
        buffer_size = 1
        batch_size = 1
        self._n_iter = len(dataset)
        self._dims_chunk = dims_chunk
        self._dims_pin = dims_pin
        self._data = util.data.MultiFileDataProvider(dataset, 1, self._n_iter, batch_size, replace_interval=1,
                                                     dims_chunk=self._dims_chunk,
                                                     dims_pin=self._dims_pin)

    def get_dims_chunk(self):
        return self._data.get_dims_chunk()

    def __len__(self):
        return self._n_iter

    def __iter__(self):
        return self._data