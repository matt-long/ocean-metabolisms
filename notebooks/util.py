import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def attrs_label(attrs): 
    """generate a label from long_name and units"""
    return f'{attrs["long_name"]} [{attrs["units"]}]'

