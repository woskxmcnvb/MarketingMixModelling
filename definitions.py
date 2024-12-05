import pandas as pd
import numpy as np

# spec X keys
Y = 'y'
MEDIA_OWN = 'media_own'
MEDIA_OWN_LOW_RET = "media_own_low_retention"
MEDIA_COMP = 'media_competitors'
PRICE = 'price'
WSD = 'wsd'
BRAND = 'brand'
STRUCT = 'structural'
FOURIER_SEASONALITY = 'fourier_seasonality'


# model sites
MODEL_BASE = 'base'
MODEL_SEASONAL = 'seasonal'
MODEL_MEDIA = 'media'
MODEL_Y = 'y'



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


