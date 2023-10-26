import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from  explainerdashboard.custom import *
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

from tabs import ModelSummaryTab, PredictionsTab


SHOPPERS_DATASET = 'https://raw.githubusercontent.com/ErmakovaAna/shoppers-intention-prediction/main/EDA/shoppers_preprocessed.csv'
df = pd.read_csv(SHOPPERS_DATASET)

X = df.drop('Revenue', axis=1)
y = df['Revenue']

numerical = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
             'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',
             'PageValues', 'SpecialDay']

categorical = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
               'VisitorType', 'Weekend']

ct = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('scaling', MinMaxScaler(), numerical)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

X_train_transformed = ct.fit_transform(X_train)
X_test_transformed = ct.transform(X_test)
new_features = list(ct.named_transformers_['ohe'].get_feature_names_out())
new_features.extend(numerical)
X_train_transformed = pd.DataFrame.sparse.from_spmatrix(X_train_transformed, columns=new_features)
X_test_transformed = pd.DataFrame.sparse.from_spmatrix(X_test_transformed, columns=new_features)

model = RandomForestClassifier(class_weight='balanced',
                               criterion='entropy',
                               max_depth=15,
                               max_features='sqrt',
                               n_estimators=500
                            ).fit(X_train_transformed, y_train)

explainer = ClassifierExplainer(
    model,
    X_test_transformed, y_test,
    labels=['No Purchase', 'Purchase']
)

db = ExplainerDashboard(explainer, [ModelSummaryTab, PredictionsTab],
                        title='Online Shoppers Intention Prediction',
                        hide_header=True,
                        whatif=False,
                        shap_interaction=False,
                        # no_permutation=True,
                        decision_trees=False,
                        header_hide_selector=True,
                        bootstrap=dbc.themes.FLATLY)

db.to_yaml('dashboard.yaml', explainerfile='explainer.joblib', dump_explainer=True)
