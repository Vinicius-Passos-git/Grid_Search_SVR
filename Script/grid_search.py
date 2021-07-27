#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

print('Bibliotecas importadas!')
# %%
df = pd.read_csv('housing.csv')

print('Dados carregados!')
# %%
#A criação do train_test será 'proporcional'

df["income_cat"] = pd.cut(df["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

print('Dados de treino criado com sucesso!')
# %%
X_train = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
y_train = strat_train_set["median_house_value"].copy()

print('Dados de treino copiado com sucesso!')
# %%
#Função para criar novas variaveis

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

print('Função criada com sucesso')
# %%
#pipeline para processar os dados e montar o modelo

num_attribs = list(X_train.drop("ocean_proximity", axis=1))
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

num_attribs = list(X_train.drop("ocean_proximity", axis=1))
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

model = Pipeline([
        ("preparation", full_pipeline),
        ("svm_reg", SVR())
    ])

print('Pipelines criado com sucesso!')
# %%
#Criação do Grid-Search

param_grid = [
        {'svm_reg__kernel': ['linear'], 'svm_reg__C':[10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'svm_reg__kernel': ['rbf'], 'svm_reg__C':[1.0, 3.0, 10., 30., 100., 300., 1000.0],
        'svm_reg__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
        ]

grid_search = GridSearchCV(model, 
                param_grid,
                cv = 5,
                scoring='neg_mean_squared_error',
                verbose = 2)

print('Grid_Search criado!')
# %%
grid_search.fit(X_train, y_train)

# %%
grid_search.best_params_

# %%
negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
# %%
