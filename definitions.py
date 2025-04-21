import pandas as pd
import jax.numpy as jnp

from .smoother import Smoother

# spec X keys
MEDIA_OWN = 'media_own'
MEDIA_COMP = 'media_competitors'

# chart spec format 
decomposition_spec = {
    "Base level": 'base',
    'Own media': 'Own media', 
    'Competitors media': 'Competitors media',
    'Non-media': [
            (          'Pricing',            'Price long'),
            (          'Pricing',           'Price short'),
            ( 'Other structural',                 'Brand'),
            ( 'Other structural',                'Demand'),
            ( 'Other structural',                   'WSD')
    ]
}

media_spec = {
    "Own media": {
        "Own media": ['CH TV', 'CH OLV', 'CH OOH', 'CH RADIO', 'CH SP&BLOGG', 'CH BANN', 'CH ECOM'],
    },
    "Competitors media": {
         "Competitors media": ['Compets Actimuno', 'Compets other'], 
    }
}



class VariableGroup:
    spec: dict = None
    #data: np.array = None
    data_index: jnp.array
    ref_to_inputs = None
    
    def __init__(self, ref_to_inputs):
        self.data_index = None
        self.ref_to_inputs = ref_to_inputs

    
    def SetDataIndex(self, _from, _to):
        self.data_index = jnp.arange(_from, _to)

    def CheckFixSpec(self, spec): 
        for field in ["name", "type", "variables"]:
            assert field in spec, "Missing key '{}' in {}".format(field, spec["name"])
        
        assert isinstance(spec["name"], str), "Wrong 'name' format in {}".format(spec["name"])
        
        assert spec["type"] in ["media", "non-media"], "Wrong 'type' {} in {}".format(spec["type"], spec["name"])
        
        if spec["type"] == "non-media":
            assert "scaling" in spec, "Missing 'scaling' key in {}".format(spec["name"])
        if "scaling" in spec:
            assert spec["scaling"] in ["total", "column", None],\
                "Wrong 'scaling' {} in {}".format(spec["scaling"], spec["name"])
            
        if "scaling max" in spec:
            assert isinstance(spec["scaling max"], int), "Wrong 'scaling max' value in {}".format(spec["name"])
        else: 
            spec["scaling max"] = None

        if spec["type"] == "media":
            if "saturation" not in spec:
                print("WARNING! 'saturation' key not in spec for {}. Setting to False, not used.".format(spec["name"]))
                spec["saturation"] = False
            assert spec["saturation"] in ['global', 'local', False], "Wrong 'saturation' value in {}".format(spec["name"])

            if "global retention" in spec.keys():
                assert isinstance(spec["global retention"], tuple) or isinstance(spec["global retention"], list) or spec["global retention"] == False,\
                    "Wrong 'global retention' value in {}. list or Touple or false is expected".format(spec["name"])
            else: 
                spec["global retention"] = False

            if "long-term effect" in spec.keys():
                assert isinstance(spec["long-term effect"], bool), "Wrong 'long-term effect' value in {}".format(spec["name"])
                if (spec["long-term effect"] == True):
                    if spec["global retention"] == False:
                        print("WARNING! Local retentions is not supported together with 'long-term effect'. For {} setting 'global retention' to (3, 1).".format(spec["name"]))
                        spec["global retention"] = (3, 1)
                    if spec["saturation"] == 'local':
                        print("WARNING! Local saturations is not supported together with 'long-term effect'. For {} setting 'saturation' to 'global'.".format(spec["name"]))
                        spec["saturation"] = 'global'
            else: 
                spec["long-term effect"] = False

        assert isinstance(spec["variables"], list), "Wrong 'variables' format in {}".format(spec["name"])
        keep_vars = [self.__CheckSingleVariableSpec(var, spec) for var in spec["variables"]]
        spec["variables"] = [var for keep, var in zip(keep_vars, spec["variables"]) if keep]
        
        return spec

    def __CheckData(self, data: pd.DataFrame) -> bool: 
        assert self.spec is not None, "Initiate first"
        for v in self.VarColumns():
            assert v in data.columns, "Missing column {}".format(v)
        return True
    
    def __CheckSingleVariableSpec(self, var, spec) -> bool:
        assert isinstance(var, dict), "Variable format is not dict in {}".format(spec["name"])
        
        if "---" in var: 
            return False

        for field in ["name", "column"]:
            assert field in var, "Missing key '{}' in {}".format(field, spec["name"])
        
        assert isinstance(var["name"], str), "Wrong 'name' format in {}".format(spec["name"])
        assert isinstance(var["column"], str), "Wrong 'column' format in {}".format(spec["name"])

        if spec["type"] == "media":
            assert "rolling" in var, "Missing 'rolling' key in {}".format(spec["name"])
            assert isinstance(var["rolling"], int) and var["rolling"] > 0, "Wrong 'rolling' value in {}".format(spec["name"])
        
            if spec["global retention"] == False:
                assert "retention" in var, "Missing 'retention' key in {}".format(spec["name"])
                assert (isinstance(var["retention"], tuple) or isinstance(var["retention"], list)) and len(var["retention"]) == 2,\
                    "Wrong 'retention' value in {}".format(spec["name"])
        return True

    def FromDict(self, spec: dict):
        self.spec = self.CheckFixSpec(spec)
        return self 
    
    def Name(self) -> str:
        return self.spec["name"]

    def Type(self):
        return self.spec["type"]
    
    def Dims(self) -> int: 
        # int - количество переменных 
        return len(self.spec["variables"])
    
    def Saturation(self) -> str | bool: 
        return self.spec["saturation"]
    
    def GlobalRetention(self) -> bool: 
        return self.spec["global retention"]
    
    def HasLongTermEffect(self) -> bool:
        return self.spec["long-term effect"]
    
    def VarNames(self): 
        assert self.spec is not None, "Initiate first"
        return [v["name"] for v in self.spec["variables"]]
    
    def VarNamesAsTuples(self, suffix: str = "") -> tuple: 
        assert self.spec is not None, "Initiate first"
        return [(self.Name() + suffix, v["name"]) for v in self.spec["variables"]]
    
    def VarColumns(self):
        assert self.spec is not None, "Initiate first"
        return [v["column"] for v in self.spec["variables"]]
    
    def ForceVector(self):
        assert self.spec is not None, "Initiate first"
        return jnp.array([v["force_positive"] for v in self.spec["variables"]])
    
    def BetaVector(self):
        assert self.spec is not None, "Initiate first"
        return jnp.array([v["beta"] for v in self.spec["variables"]])
    
    def RetentionVector(self):
        assert self.spec is not None, "Initiate first"
        assert self.spec["global retention"] == False, "Global retention settings"
        return jnp.array([v["retention"][0] for v in self.spec["variables"]]), jnp.array([v["retention"][1] for v in self.spec["variables"]])
    
    def RollingDict(self): 
        # если четное делает +1 
        assert self.spec is not None, "Initiate first"
        def _to_odd(x):
            return x + 1 if x % 2 == 0 else x
        return {v["column"]: _to_odd(v["rolling"]) for v in self.spec["variables"]}
    
    def PrepareNonMediaData(self, df: pd.DataFrame):
        assert self.__CheckData(df)
        return Smoother().Impute(
            Scaler(
                scaling='min_max', 
                scaler_from=self.spec["scaling"], 
                centering='first'
            ).FitTransform(df[self.VarColumns()])
        )
    
    def PrepareMediaData(self, df: pd.DataFrame) -> jnp.array:
        assert self.__CheckData(df)
        return df[self.VarColumns()]\
            .where(df[self.VarColumns()] > 0)\
            .apply(lambda x: x.rolling(window=self.RollingDict()[x.name], min_periods=1, center=True).mean())\
            .where(df[self.VarColumns()] > 0)\
            .fillna(0)\
            .pipe(lambda x: x.div(x.max(axis=None) if self.spec["scaling max"] is None else self.spec["scaling max"]))\
            .values
    
    def GetData(self) -> jnp.array:
        if self.Type() == 'media':
            return self.ref_to_inputs.media_data[:, self.data_index]
        elif self.Type() == 'non-media':
            return self.ref_to_inputs.non_media_data[:, self.data_index]
        else: 
            raise ValueError("VarGroup: GetData")

    


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
        print("min: {}, max: {}, scaling: {}, centeing: {}".format(self.min_, self.max_, self.scaling, self.centering))

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