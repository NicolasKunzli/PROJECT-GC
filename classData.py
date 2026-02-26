from simbarca_base import SimbarcaBase, SimBarcaForecast
import numpy as np

class NewSim(SimBarcaForecast):

    def __init__(self, split = "train"):
        super().__init__(split)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        return 0

    def load_or_compute_metadata(self):
        return None
    
NewSims = NewSim(split="test")

a = NewSims.get_session_properties()
print(len(a["session_ids"]))
