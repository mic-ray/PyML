import pandas as pd
from sklearn.datasets import load_wine

wines = load_wine(as_frame=True)
print(wines)
