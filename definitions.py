import pandas as pd
import numpy as np

from .smoother import Smoother

# spec X keys
MEDIA_OWN = 'media_own'
MEDIA_COMP = 'media_competitors'

# spec dict format 
spec_new = {
    "media": [
        {
            "name": "Own media",
            "scaling": "total",
            "saturation": True,
            "variables": [
                {"name": "TV", "column": "CH TV", 
                 "rolling": 3, "retention": (3, 1), "beta": 1, "force_positive": True},
                {"name": "OLV",   "column": "CH OLV", 
                 "rolling": 3, "retention": (3, 1), "beta": 1, "force_positive": True},
                {"name": "OOH", "column": "CH OOH", 
                 "rolling": 3, "retention": (3, 1), "beta": 1, "force_positive": True},
                {"name": "Radio",    "column": "CH RADIO", 
                 "rolling": 3, "retention": (3, 1), "beta": 1, "force_positive": True},
                {"name": "Projects & bloggers", "column": "CH SP&BLOGG", 
                 "rolling": 3, "retention": (3, 1), "beta": 1, "force_positive": True},
                
                {"name": "Banners", "column": "CH BANN", 
                 "rolling": 1, "retention": (1, 3), "beta": 1, "force_positive": True},
                {"name": "E-com", "column": "CH ECOM", 
                 "rolling": 1, "retention": (1, 3), "beta": 1, "force_positive": True},
            ]
        }, 
        {
            "name": "Competitors media",
            "scaling": "total",
            "saturation": False,
            "variables": [
                {"name": "Actimuno ads", "column": "Compets Actimuno", 
                 "rolling": 3, "retention": (3, 1), "beta": 1, "force_positive": False},
                {"name": "Other competitors ads", "column": "Compets other", 
                 "rolling": 3, "retention": (3, 1), "beta": 1, "force_positive": False},
            ]
        }
    ],  
    "non_media": [
        {
            "name": "Pricing",
            "scaling": "total",
            "variables": [
                {"name": "Price long", "column": "LongPriceIndex", "beta": 1, "force_positive": False}, 
                {"name": "Price short", "column": "ShortPriceIndex", "beta": 1, "force_positive": False},
            ]
        },
        {
            "name": "Other structural",
            "scaling": "column",
            "variables": [
                {"name": "Brand", "column": "Brand modeled", "beta": 1, "force_positive": True}, 
                {"name": "Demand", "column": "Demand", "beta": 1, "force_positive": False},
                {"name": "SVO", "column": "SVO", "beta": 1, "force_positive": False},
                {"name": "WSD", "column": "WSD", "beta": 1, "force_positive": True},
            ]
        },
    ]
}



class VariableGroup:
    spec: dict = None
    data: np.array = None
    
    def __init__(self):
        pass

    def CheckSpec(self, spec): 
        for field in ["name", "type", "variables"]:
            assert field in spec, "Missing key '{}' in {}".format(field, spec["name"])
        
        assert isinstance(spec["name"], str), "Wrong 'name' format in {}".format(spec["name"])
        
        assert spec["type"] in ["media", "non-media"], "Wrong 'type' {} in {}".format(spec["type"], spec["name"])
        
        if spec["type"] == "non-media":
            assert "scaling" in spec, "Missing 'scaling' key in {}".format(spec["name"])
        if "scaling" in spec:
            assert spec["scaling"] in ["total", "column"], "Wrong 'scaling' {} in {}".format(spec["scaling"], spec["name"])

        if spec["type"] == "media":
            assert "saturation" in spec, "Missing 'saturation' key in {}".format(spec["name"])
            assert isinstance(spec["saturation"], bool), "Wrong 'saturation' format in {}".format(spec["name"])

        assert isinstance(spec["variables"], list), "Wrong 'variables' format in {}".format(spec["name"])
        for var in spec["variables"]:
            self.__CheckSingleVariableSpec(var, spec)

    def __CheckData(self, data: pd.DataFrame) -> bool: 
        assert self.spec is not None, "Initiate first"
        for v in self.VarColumns():
            assert v in data.columns, "Missing column {}".format(v)
        return True
    
    def __CheckSingleVariableSpec(self, var, spec):
        assert isinstance(var, dict), "Variable format is not dict in {}".format(spec["name"])
        for field in ["name", "column"]:
            assert field in var, "Missing key '{}' in {}".format(field, spec["name"])
        
        assert isinstance(var["name"], str), "Wrong 'name' format in {}".format(spec["name"])
        assert isinstance(var["column"], str), "Wrong 'column' format in {}".format(spec["name"])

        if spec["type"] == "media":
            assert "rolling" in var, "Missing 'rolling' key in {}".format(spec["name"])
            assert isinstance(var["rolling"], int) and var["rolling"] > 0, "Wrong 'rolling' value in {}".format(spec["name"])
        
        if spec["type"] == "media":
            assert "retention" in var, "Missing 'retention' key in {}".format(spec["name"])
            assert isinstance(var["retention"], tuple) and len(var["retention"]) == 2, "Wrong 'retention' value in {}".format(spec["name"])

    def FromDict(self, spec: dict):
        self.CheckSpec(spec)
        self.spec = spec
        return self 
    
    def Name(self):
        return self.spec["name"]

    def Type(self):
        return self.spec["type"]
    
    def Dims(self) -> int: 
        return len(self.spec["variables"])
    
    def Saturation(self) -> bool: 
        return self.spec["saturation"]
    
    def VarNames(self): 
        assert self.spec is not None, "Initiate first"
        return [v["name"] for v in self.spec["variables"]]
    
    def VarNamesAsTuples(self): 
        assert self.spec is not None, "Initiate first"
        return [(self.Name(), v["name"]) for v in self.spec["variables"]]
    
    def VarColumns(self):
        assert self.spec is not None, "Initiate first"
        return [v["column"] for v in self.spec["variables"]]
    
    def ForceVector(self):
        assert self.spec is not None, "Initiate first"
        return np.array([v["force_positive"] for v in self.spec["variables"]])
    
    def BetaVector(self):
        assert self.spec is not None, "Initiate first"
        return np.array([v["beta"] for v in self.spec["variables"]])
    
    def RetentionVector(self):
        assert self.spec is not None, "Initiate first"
        return np.array([v["retention"][0] for v in self.spec["variables"]]), np.array([v["retention"][1] for v in self.spec["variables"]])
    
    def RollingDict(self): 
        # если четное делает +1 
        assert self.spec is not None, "Initiate first"
        def _to_odd(x):
            return x + 1 if x % 2 == 0 else x
        return {v["column"]: _to_odd(v["rolling"]) for v in self.spec["variables"]}
    
    def PrepareData(self, df: pd.DataFrame):
        assert self.spec is not None, "Initiate first"
        assert self.__CheckData(df)

        if self.spec["type"] == "media":
            rolling_dict = self.RollingDict()
            self.data = df[self.VarColumns()]\
                    .where(df[self.VarColumns()] > 0)\
                    .apply(lambda x: x.rolling(window=rolling_dict[x.name], min_periods=1, center=True).mean())\
                    .where(df[self.VarColumns()] > 0)\
                    .fillna(0)\
                    .pipe(lambda x: x.div(x.max(axis=None)))\
                    .values
        elif self.spec["type"] == "non-media":
            self.data = Smoother().Impute(
                Scaler(
                    scaling='min_max', 
                    scaler_from=self.spec["scaling"], 
                    centering='first').FitTransform(df[self.VarColumns()])
            )
        return self
    
    def GetData(self):
        return self.data
    


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

    def __init__(self, scaling='min_max', scaler_from='column', centering=None):
        # scaling in 'min_max' / 'max_only'
        assert scaler_from in ['total', 'column']
        assert scaling in ['min_max', 'max_only']
        assert centering in [None, 'mean', 'first']
        self.scaling = scaling
        self.centering = centering
        self.scaler_from = scaler_from

    def Inspect(self):
        print("min: {}, max: {}, scaling: {}, centeing: {}".format(self.min_, self.max_, self.scaling, self.centering))

    def Fit(self, data: pd.DataFrame):
        self.fit_shape_ = data.shape
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