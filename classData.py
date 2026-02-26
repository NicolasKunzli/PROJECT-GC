from simbarca_base import SimbarcaBase, SimBarcaForecast
import numpy as np

class NewSim(SimBarcaForecast):
    """
    We define the abstract methods needed for the other classes
    
    By calling SimBarcaForecast, we also call SimBarcaBase, which also calls BaseDataset
    """
    def __init__(self, split = "train"):
        super().__init__(split) #super() let us access the parent class, namely SimBarcaForecast

    def __len__(self):
        """
        PLACEHOLDER
        """
        pass

    def __getitem__(self, idx):
        """
        PLACEHOLDER
        """
        return 0

    def load_or_compute_metadata(self):
        """
        PLACEHOLDER
        """
        return None
    
NewSims = NewSim(split="train")

a = NewSims.init_graph_structure()
print(a)
b = NewSims.cluster_id
print(np.shape(b))
