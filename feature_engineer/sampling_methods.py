#oversample rare, undersample dominant
import imblearn
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
print(imblearn.__version__)

# define oversampling strategy
def oversample_rare(df:pd.DataFrame,target:str,x_vars:list):
    over = RandomOverSampler(sampling_strategy='minority')

    # fit and apply the transform
    X, y = over.fit_resample(np.array(df[x_vars]), df[[target]])
    return X,y


# define oversampling strategy
def undersample_common(df: pd.DataFrame, target: str, x_vars: list):
    under = RandomUnderSampler()

    # fit and apply the transform
    X, y = under.fit_resample(np.array(df[x_vars]), df[[target]])
    return X, y
