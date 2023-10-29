import numpy as np

from ..register import register, DIST_LAB_TAB

@register(DIST_LAB_TAB)
class DistLabelTable(object):
    def __init__(self, eps=1e-6):
        super().__init__()
        
        self.eps = eps
        self.count = 0
        self.dist_array = None
        self.inv_dist_indices = None

    def dist(self):
        return self.dist_array

    def inv_dist_idx(self):
        return self.inv_dist_indices

    def dist_2_inv_idx(self, dist):
        raise NotImplementedError()

    def inv_idx_2_dist(self, indices):
        raise NotImplementedError()

    def __len__(self):
        return self.count

    def get_dist(self, idx):
        if idx < 0 or idx >= len(self):
            raise Exception(f'idx = {idx}, len(self) = {len(self)}')

        return self.dist_array[idx]

    def get_inv_dist_idx(self, idx):
        if idx < 0 or idx >= len(self):
            raise Exception(f'idx = {idx}, len(self) = {len(self)}')

        return self.inv_dist_indices[idx]

@register(DIST_LAB_TAB)
class DummyDisparityDistLabelTable(DistLabelTable):

    def __init__(self, bf, eps=1e-6):
        super().__init__(eps=eps)

        self.bf = bf

    def dist_2_inv_idx(self, dist):
        return self.bf / ( dist + self.eps )

    def inv_idx_2_dist(self, indices):
        return self.bf / ( indices + self.eps )

@register(DIST_LAB_TAB)
class LinspaceDistLabelTable(DummyDisparityDistLabelTable):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            max_index=192,
            num_stops=24,
            bf=96,
            eps=1e-6 )

    def __init__(self,
        max_index, # One past last.
        num_stops,
        bf,
        eps=1e-6):
        super().__init__(bf=bf, eps=eps)

        self.inv_dist_indices = \
            np.linspace( 0, max_index, num_stops ).astype(np.float32)
        self.dist_array = self.inv_idx_2_dist( self.inv_dist_indices )
        self.count = num_stops

@register(DIST_LAB_TAB)
class LogspaceDistLabelTable(DummyDisparityDistLabelTable):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            max_index=192,
            num_stops=24,
            bf=96,

            eps=1e-6 )

    def __init__(self,
        max_index, # One past last.
        num_stops,
        bf,
        eps=1e-6):
        super().__init__(bf=bf, eps=eps)

        self.inv_dist_indices = \
            np.geomspace( 1, max_index, num_stops).astype(np.float32)
        self.dist_array = self.inv_idx_2_dist( self.inv_dist_indices )
        self.count = num_stops

@register(DIST_LAB_TAB)
class FixedDistLabelTable(DummyDisparityDistLabelTable):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            dist_list=[0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100],
            bf=96,
            eps=1e-6 )

    def __init__(self,
        dist_list,
        bf,
        eps=1e-6):
        super().__init__(bf=bf, eps=eps)

        self.dist_array = np.array(dist_list, dtype=np.float32)
        self.inv_dist_indices = self.dist_2_inv_idx( self.dist_array )
        self.count = len(dist_list)
        