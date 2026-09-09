from pathlib import Path

from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import pandas as pd

# resolve the csvs relative to this file so imports work from any working dir
_DATA_DIR = Path(__file__).resolve().parent

numerical_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km']
nominal_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Number_of_Cars_Owned', 'Environmental_Concern_Level', 'Range_Anxiety_Level']
scalar_cols = ['Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Buy_Score']


def load_n_clean():
    Xtr = pd.read_csv(_DATA_DIR / 'train.csv')
    Xte = pd.read_csv(_DATA_DIR / 'test.csv')
    ytr = Xtr.pop("Will_Buy_EV")
    test_ids = Xte['id']
    Xtr['Home_Charging_Possible'] = Xtr['Home_Charging_Possible'].map({'Yes': 1, 'No': 0})
    Xte['Home_Charging_Possible'] = Xte['Home_Charging_Possible'].map({'Yes': 1, 'No': 0})
    Xtr['Subsidy_Available'] = Xtr['Subsidy_Available'].map({'Yes': 1, 'No': 0})
    Xte['Subsidy_Available'] = Xte['Subsidy_Available'].map({'Yes': 1, 'No': 0})
    ytr = ytr.map({'Yes': 1, 'No': 0})
    Xtr.drop(columns='id', inplace=True)
    Xte.drop(columns='id', inplace=True)
    Xtr['income_is_30k'] = (Xtr['Annual_Income_USD'] == 30000).astype(int)
    Xte['income_is_30k'] = (Xte['Annual_Income_USD'] == 30000).astype(int)
    Xtr['Daily_Commute_is_5km'] = (Xtr['Daily_Commute_km'] == 5).astype(int)
    Xte['Daily_Commute_is_5km'] = (Xte['Daily_Commute_km'] == 5).astype(int)
    Xtr['Charging_Stations_Near_Home'] = Xtr['Charging_Stations_Near_Home'].clip(upper=7)
    Xte['Charging_Stations_Near_Home'] = Xte['Charging_Stations_Near_Home'].clip(upper=7)
    Xtr['Charging_Stations_Near_Work'] = Xtr['Charging_Stations_Near_Work'].clip(upper=9)
    Xte['Charging_Stations_Near_Work'] = Xte['Charging_Stations_Near_Work'].clip(upper=9)
    anxiety_map = {'Low': 0.0, 'Medium': 1.0, 'High': 3.0}
    Xtr['Buy_Score'] = 1.2 * (Xtr['Annual_Income_USD'] / 100000) + 0.6 * Xtr['Environmental_Concern_Level'] + 2 * Xtr['Subsidy_Available'] - Xtr['Range_Anxiety_Level'].map(anxiety_map).astype('float32')
    Xte['Buy_Score'] = 1.2 * (Xte['Annual_Income_USD'] / 100000) + 0.6 * Xte['Environmental_Concern_Level'] + 2 * Xte['Subsidy_Available'] - Xte['Range_Anxiety_Level'].map(anxiety_map).astype('float32')
    # a category outside the maps above (Yes/No, anxiety) would silently become
    # NaN and flow into Buy_Score / the scaler - fail loudly instead
    _mapped = ['Home_Charging_Possible', 'Subsidy_Available', 'Buy_Score']
    assert not Xtr[_mapped].isna().any().any() and not Xte[_mapped].isna().any().any()
    return Xtr, Xte, ytr, test_ids



def make_preprocessor():
    return ColumnTransformer([('cont', PowerTransformer(), numerical_cols),
                            ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_cols),
                            ('scalar', StandardScaler(), scalar_cols)],
                            remainder='passthrough',
    )
