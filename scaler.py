import pandas as pd


class Scaler:
    """
    scaling in ['min_max', 'max_only']
    mode in ['total', 'column']
    centering in [None, 'mean', 'first']
    """
    min_ = None
    max_ = None
    centering_shift = 0
    fit_shape_ = None
    scaling: str
    centering: str = None
    scaler_from: str

    def __init__(self, scaling='min_max', scaler_from=None, centering=None, min_max_limts=None):
        # scaling in 'min_max' / 'max_only'
        assert scaler_from in ['total', 'column', None]
        assert scaling in ['min_max', 'max_only']
        assert centering in [None, 'mean', 'first']
        assert min_max_limts is None or isinstance(min_max_limts, list) or isinstance(min_max_limts, tuple),\
            "Scaler init: Wrong min_max_limts value"
        if scaler_from is None:
            assert min_max_limts is not None, "Either scaler_from ({}) or min_max_limts ({}) must be provided".format(scaler_from, min_max_limts)

        self.scaling = scaling
        self.centering = centering
        self.scaler_from = scaler_from
        if min_max_limts:
            self.min_, self.max_ = min_max_limts

    def Inspect(self):
        print("min: {}, max: {}, mode: {}, source: {}, centeing: {}".format(self.min_, self.max_, self.scaling, self.scaler_from, self.centering))

    def Fit(self, data: pd.DataFrame):
        self.fit_shape_ = data.shape
        if (self.max_ is None) or (self.min_ is None):
            axis_ = 0 if self.scaler_from == 'column' else None
            self.min_ = data.min(axis=axis_) if self.scaling == 'min_max' else 0
            self.max_ = data.max(axis=axis_)
        if self.centering:
            transformed = self.Transform(data)
            if self.centering == 'first':
                self.centering_shift = transformed.iloc[0]
            elif self.centering == 'mean':
                self.centering_shift = transformed.mean(axis=0)
        return self 

    def Transform(self, data: pd.DataFrame):
        #assert data.shape == self.fit_shape_
        return data.sub(self.min_).div(self.max_ - self.min_).sub(self.centering_shift)
    
    def FitTransform(self, data: pd.DataFrame):
        return self.Fit(data).Transform(data)
    
    def InverseTransform(self, data):
        #assert data.shape == self.fit_shape_
        return data.add(self.centering_shift).mul(self.max_ - self.min_).add(self.min_)