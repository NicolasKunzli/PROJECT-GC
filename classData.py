from simbarca_base import SimbarcaBase, SimBarcaForecast
import numpy as np

class NewSim(SimBarcaForecast):

    def __init__(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...

    def load_or_compute_metadata(self):
        ...
    
dataset = NewSim(split="train")
