import pandas as pd
import numpy as np
from scipy import stats
import csv
import sys

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import econml
from econml.metalearners import XLearner

import warnings
from numba.core.errors import NumbaDeprecationWarning


def main():
    args = sys.argv[1:]
    T_index, Y_index, Val_index = 0, 1, 2
    # Treatment Type
    if args[T_index] == 'Status':
        
    else:
        Y_index += 1
        Val_index += 1
        if args[1] == 'Any':
            accumulate = np.any
        elif args[1] == 'All':
            accumulate = np.all
        elif args[1] == 'Mode':
            accumulate = stats.mode
        else:
            raise ValueError("Unknown accumulation method")
    
    # Outcome Type
    if args[Y_index] == 'Conversion':
        pass
    elif args[Y_index] == 'Frequency':
        pass
    elif args[Y_index] == 'Monetary':
        pass
        
    # Validation Type
    if args[Val_index] == 'PredefinedHyperparameters':
        pass
    elif args[Val_index] == 'FastValidation':
        pass
    elif args[Val_index] == 'FullValidation':
        pass
    
    print('STARTED')
    max_clients, tol = None, 0
    df = pd.read_csv('data.csv', sep=';', parse_dates=['Дата'], dayfirst=True)
    df.rename(columns={"Дата": "Date", "Профиль участника": "UID", "Вид операции": "OperationType", "Сумма": "Price",
                       "Сумма списанных монет": "SpentCoins", "Сумма начисленных монет": "GainedCoins",
                       "Ресторан": "Point", "Агент продаж": "Agent", "Статус": "Status"}, inplace=True)
    df.set_index('Date', inplace=True)
    df.drop(columns=['Номер'], inplace=True)
    day_of_week_labels = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    months_labels = ['Январь', 'Февраль', 'Май', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь',
                     'Ноябрь', 'Декабрь']
    assert (len(day_of_week_labels) == 7 and len(months_labels) == 12)

    start_dt, end_dt = df.iloc[0].name, df.iloc[-1].name
    print(f'Датасет начинается {start_dt.day} {months_labels[start_dt.month - 1][:-1]}я и '
          f'заканчивается {end_dt.day} {months_labels[end_dt.month - 1][:-1]}я включительно')
    m, n = df.shape
    numerical = ['Price', 'SpentCoins', 'GainedCoins', 'ActualPrice']
    categorical = ['OperationType', 'Point', 'Agent', 'Status']
    print(f"В датасете {m} строк и {n} столбцов")
    print(f"Из них {len(numerical)} численных и {len(categorical)} категориальных")
    print(df.columns)
    assert (len(numerical) + len(categorical) == n)
    cols_with_nans = []
    for col in df:
        if df[col].isna().sum() > 0:
            cols_with_nans.append(col)
            df[col].fillna(0, inplace=True)
    print("Найдены пропуски в колонках", *cols_with_nans)
    print("Пропуски заполнены нулями")
    
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Hour'] = df.index.hour
    df['DayOfYear'] = df.index.dayofyear

    df['Discount'] = df['Status'].replace(
        {"Бронзовый ключ": 0.02, "Серебряный ключ": 0.03, "Золотой ключ": 0.05, "Платиновый ключ": 0.07,
         "Сотрудник": 0.15})
    clients = df.groupby('UID').Status.last().to_frame()
    clients['FavouriteDay'] = df.groupby(['UID', 'DayOfWeek']).Price.mean().to_frame().reset_index().sort_values(
        by=['UID', 'Price'], ascending=False).groupby('UID').DayOfWeek.first()
    hierarchy = {"Бронзовый ключ": 0, "Серебряный ключ": 1, "Золотой ключ": 2, "Платиновый ключ": 3, "Сотрудник": 4}
    df['StatusNew'] = df["Status"].apply(lambda x: hierarchy[x])
    middle_data = df.loc['2023-02-15':'2023-03-15'].groupby('UID').StatusNew  # TODO
    clients['TreatmentStatus'] = (middle_data.is_monotonic_increasing) & (middle_data.nunique() > 1)
    clients['TreatmentStatus'].fillna(value=False, inplace=True)
    clients['TreatmentStatus'] = clients['TreatmentStatus'].astype(int)
    df['SberSpasibo'] = 0
    df.loc[df.Price - df.GainedCoins / df.Discount > 10, 'SberSpasibo'] = 1
    clients['SberSpasibo'] = df.groupby('UID').SberSpasibo.any()
    clients['first_part'] = df.loc[:'2023-03-01'].groupby('UID').Price.sum()  # TODO
    clients['first_part'].fillna(0, inplace=True)
    clients['second_part'] = df.groupby('UID').Price.sum() - clients['first_part']
    clients['conversion'] = clients['second_part'] - clients['first_part']
    clients['Recency'] = df.loc['2023-02-15':].groupby('UID').DayOfYear.max()  # TODO
    clients['Frequency'] = df.loc['2023-02-15':].groupby('UID').Price.count()  # TODO
    clients['Monetary_Sum'] = df.loc['2023-02-15':].groupby('UID').Price.sum()  # TODO
    clients['Monetary_Mean'] = df.loc['2023-02-15':].groupby('UID').Price.mean()  # TODO
    clients['outcome'] = df.loc['2023-03-15':].groupby('UID').Price.sum()  # TODO
    clients['outcome'].fillna(0, inplace=True)
    clients.dropna(inplace=True)
    print(clients.shape)
    feature_names = ['Recency', 'Frequency', 'Monetary_Sum', 'Monetary_Mean']
    X, y, treatment = clients[feature_names + ['FavouriteDay', 'SberSpasibo']], clients['outcome'], clients[
        'TreatmentStatus']

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore'), ['FavouriteDay', 'SberSpasibo']),
        ('scaling', StandardScaler(), feature_names),
    ])

    X = column_transformer.fit_transform(X)

    best_params_econml = {'X_XGBoost': {'max_depth': 40, 'n_estimators': 200, 'lr': 0.01}}

    learner_x_xgboost = XLearner(models=XGBRegressor(n_estimators=best_params_econml['X_XGBoost']['n_estimators'],
                                                     max_depth=best_params_econml['X_XGBoost']['max_depth'],
                                                     learning_rate=best_params_econml['X_XGBoost']['lr']))
    print(102)
    learner_x_xgboost.fit(X=X, T=treatment, Y=y)
    print(104)
    cate_x_xgboost = learner_x_xgboost.effect(X)
    print(106)
    clients['CATE'] = cate_x_xgboost
    receptive_clients = clients[clients['CATE'] > tol]['CATE']
    print(109)
    receptive_clients.to_csv('receptive_clients.csv')
    print(110)


if __name__ == "__main__":
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    main()
    
    