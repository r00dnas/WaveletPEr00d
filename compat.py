# compat.py
try:
    import torch_sparse
except Exception as e:
    torch_sparse = None
    print("torch_sparse disabled:", e)

# optional: keep TDC off unless you need RDKit
try:
    from tdc.generation import MolGen
    from tdc.chem_utils import MolConvert
except Exception as e:
    print("TDC disabled:", e)
    class MolConvert:
        def smi2mol(self, x): return x
        def mol2smi(self, x): return x
        def smi2img(self, *a, **k): return None
    class MolGen:
        def __init__(self,*a,**k): pass
        def get_data(self,*a,**k): return []
