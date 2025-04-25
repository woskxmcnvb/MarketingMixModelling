import pandas as pd
import jax.numpy as jnp

from copy import deepcopy

from .smoother import Smoother
from .scaler import Scaler

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
    scaler: Scaler = None
    
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
        
        # scaling
        assert "scaling" in spec, "'Scaling' must be specified in {}".format(spec["name"])
        if isinstance(spec["scaling"], int):
            spec["scaling_min_max"] = [0, spec["scaling"]]
            spec["scaling_from"] = None
        elif spec["scaling"] in ["total", "column"]:
            spec["scaling_min_max"] = None
            spec["scaling_from"] = spec["scaling"]
        else:
            raise ValueError("Wrong 'scaling' {} in {}".format(spec["scaling"], spec["name"]))


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
        
        if "name" not in var: 
            print("WARNING! No 'name' key in {}".format(var))
            return True

        assert "column" in var, "Missing 'column' key in {}".format(spec["name"])
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
        self.spec = self.CheckFixSpec(spec.copy())
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
    
    def FitTransform(self, df: pd.DataFrame, fit_scaler=True) -> jnp.array:
        assert self.__CheckData(df)
        if self.spec["type"] == 'media':
            return self._FitTransformMedia(df, fit_scaler)
        elif self.spec["type"] == 'non-media':
            return self._FitTransformNonMedia(df, fit_scaler)
        
    def Transform(self, df: pd.DataFrame) -> jnp.array:
        return self.FitTransform(df, fit_scaler=False)

    def _FitTransformNonMedia(self, df: pd.DataFrame, fit_scaler=True) -> jnp.array:
        if fit_scaler:
            self.scaler = Scaler(
                    scaling='min_max', 
                    scaler_from=self.spec["scaling"], 
                    centering='first',
                    min_max_limts=None
                ).Fit(df[self.VarColumns()])
        
        return Smoother().Impute(
            self.scaler.Transform(df[self.VarColumns()])
        ) 


    def _FitTransformMedia(self, df: pd.DataFrame, fit_scaler=True) -> jnp.array:
        rolled = df[self.VarColumns()]\
            .where(df[self.VarColumns()] > 0)\
            .apply(lambda x: x.rolling(window=self.RollingDict()[x.name], min_periods=1, center=True).mean())\
            .where(df[self.VarColumns()] > 0)\
            .fillna(0)
        
        if fit_scaler:
            self.scaler = Scaler(
                    scaling='min_max', 
                    scaler_from=self.spec["scaling_from"],
                    centering=None,
                    min_max_limts=self.spec["scaling_min_max"]
                ).Fit(rolled)
        
        return self.scaler.Transform(rolled).values
    

        
class ModelTarget:
    y: jnp.array
    scaler: Scaler

    def __init__(self):
        pass

    def Fit(self, spec: dict, df: pd.DataFrame):
        assert 'y' in spec, "ERRROR! 'y' not found in spec"
        assert isinstance(spec["y"], str), "Wrong 'y' format in spec"

        self.scaler = Scaler(scaling='max_only', scaler_from='column').Fit(df[spec["y"]])
        self.y = jnp.array(Smoother().Impute(self.scaler.Transform(df[spec["y"]])))
        return self


class ModelCovs:
    covs_len: int
    
    media_vars: list[VariableGroup]
    media_data: jnp.array 
    
    non_media_vars: list[VariableGroup]
    non_media_data: jnp.array

    seasonality: dict = None
    fixed_base: bool = False
    long_term_retention: int | tuple = 1

    def __init__(self, spec: dict):
        # вынести отсюда работу с данными
        
        self.media_vars = []
        self.non_media_vars = []
        
        # check must-keys
        missing_keys = []
        for mk in ["name", "fixed base", "long-term retention", "X"]:
            if mk not in spec:
                missing_keys.append(mk)
        assert len(missing_keys) == 0, "ERRROR! Check missing keys in spec {}.".format(missing_keys)

        # check fixed base option
        assert isinstance(spec["fixed base"], bool), "Wrong 'fixed base' value, bool is expected"
        self.fixed_base = spec["fixed base"]

        if isinstance(spec["long-term retention"], tuple) or isinstance(spec["long-term retention"], list):
            self.long_term_retention = list(spec["long-term retention"])
        elif spec["long-term retention"] == 1:
            self.long_term_retention = 1 
        else:
            raise ValueError("Wrong 'long-term retention' value, 1 or list or tuple[int, int] is expected")
        
        # check fix seasonality 
        if "seasonality" in spec:
            assert "cycle" in spec["seasonality"], "Seasonality cycle must be specified"
            period = spec["seasonality"]["cycle"]
            assert period is not None, "Seasonality period must be specified"
            assert 1 < period and period <= 52, "Wrong seasonality cycle: {}".format(period)

            assert "model" in spec["seasonality"], "Seasonality model must be specified"
            model = spec["seasonality"]["model"]
            assert model is not None, "Seasonality model must be specified"
            assert model in ['fourier', 'discrete'], "Wrong seasonality model: {}".format(model)

            self.seasonality = spec["seasonality"].copy()
            
            if model == 'fourier':
                if ("num_fouries_terms" not in self.seasonality) or (self.seasonality["num_fouries_terms"] is None): 
                    self.seasonality["num_fouries_terms"] = period // 4


        # setup X
        _media_data_index = 0
        _non_media_data_index = 0
        for var_group in spec["X"]:
            if var_group["type"] == 'media':
                self.media_vars.append(VariableGroup(self).FromDict(var_group))
                dims = self.media_vars[-1].Dims()
                self.media_vars[-1].SetDataIndex(_media_data_index, _media_data_index + dims)
                _media_data_index += dims
            if var_group["type"] == 'non-media':
                self.non_media_vars.append(VariableGroup(self).FromDict(var_group))
                dims = self.non_media_vars[-1].Dims()
                self.non_media_vars[-1].SetDataIndex(_non_media_data_index, _non_media_data_index + dims)
                _non_media_data_index += dims
            else: 
                Warning("Wrong X variable type: {}. Skipped".format(var_group["type"]))

    def FitToData(self, df: pd.DataFrame):
        self.covs_len = len(df)
        _media_data = []
        _non_media_data = []

        for v in self.media_vars:
            _media_data.append(v.FitTransform(df))
        for v in self.non_media_vars:
            _non_media_data.append(v.FitTransform(df))

        self.media_data = jnp.column_stack(_media_data) if len(_media_data) > 0 else None
        self.non_media_data = jnp.column_stack(_non_media_data) if len(_non_media_data) > 0 else None
        return self
    
    def TransformData(self, df: pd.DataFrame):
        self.covs_len = len(df)
        _media_data = []
        _non_media_data = []

        for v in self.media_vars:
            _media_data.append(v.Transform(df))
        for v in self.non_media_vars:
            _non_media_data.append(v.Transform(df))

        self.media_data = jnp.column_stack(_media_data)
        self.non_media_data = jnp.column_stack(_non_media_data)
        return self
    
    def AllMediaVarnames(self, suffix="") -> list:
        return sum([g.VarNamesAsTuples(suffix) for g in self.media_vars], [])
    
    def AllNonMediaVarnames(self, suffix="") -> list:
        return sum([g.VarNamesAsTuples(suffix) for g in self.non_media_vars], [])
    
    def HasMedia(self) -> bool:
        return len(self.media_vars) > 0
    
    def HasNonMedia(self) -> bool:
        return len(self.non_media_vars) > 0

    def Copy(self):
        return deepcopy(self)

    

