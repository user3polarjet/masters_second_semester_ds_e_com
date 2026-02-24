#----------------  скорінговий аналіз реальних даних мікрокредитування  ------------------

'''

Завдання:
1. Побудувати скорінгову оцінку
2. Проввести кластерізацію заявників
3. Виявити факти шахрайства:

Package                      Version
---------------------------- -----------
pip                          23.1
numpy                        1.23.5
pandas                       1.5.3
pyod                         1.1.0
xlrd                         2.0.1
matplotlib                   3.6.2

'''
'''
Для формування SCOR-оцінки було використано багатокритеріальну оптимізацію за нелінійною схемою компромісів.
Порогове значення обрано емпірічно для ефективного відсікання клієнтів із низькою платіжньою спроможністю.
Для перевірки адекватності відсікання багатокритеріальної SCOR-оцінки використано лінійний дискрименантний аналіз.
Пошук фальсифікації реалізовано як задача пошуку аномалій.
Це надає можливості:
- сформувати поділ на достовірні та шахрайскі дані;
- сформувати рішення про ймовірність повернення кредиту та не повернення.

Бібліотека Python для виявлення сторонніх об’єктів у багатовимірних даних - виявлення аномалій / шахрайства
https://pyod.readthedocs.io/en/latest/

'''

# ------------------------------ Imports --------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pyod.models.knn import KNN
import seaborn as sns

# -------------------------------- Data preparation (підготовка) ---------------------------------

def data_preparation (filname):
    descriptions = pd.read_excel(filname)
    client_bank_descriptions = descriptions[
        (descriptions.Place_of_definition == 'Вказує позичальник') |
        (descriptions.Place_of_definition == 'параметри, повязані з виданим продуктом')
        ]
    client_bank_fields = client_bank_descriptions["Field_in_data"]

    data = pd.read_excel("sample_data.xlsx")

    data = data.loc[:, list(set(client_bank_fields).intersection(data.columns))]
    data = data.dropna(axis=1)
    data.head()
    return data

# --------------------------- Data labelling (маркування) ------------------------------------------
def clip_data (data):
    return np.where(np.sign(data) >=0, 1, -1) * np.clip(np.abs(data), 1e-9, None)

def min_max (data):
    return (data - data.min())/ (data.max() - data.min())

def voronin (data: pd.DataFrame, weights: np.array, direction: np.array, delta=0.1) -> np.array:
    data = data.copy()
    data.loc[direction] = 1 / clip_data(data[direction].values)
    data = data.values
    criteria_sum = np.sum(data, axis=1, keepdims=True)
    normalized_criteria_values = data / clip_data(criteria_sum)
    integro = np.dot(weights, 1/(1 - normalized_criteria_values))
    return integro

# --------------------------------- Linear discriminant analysis ----------------------------
def linear_discriminant_analysis (minimax_data):
    viz_data = minimax_data.T
    compressed = LinearDiscriminantAnalysis().fit_transform(viz_data, data["give"])
    plt.figure(figsize=(10, 5))
    plt.scatter(compressed, np.zeros_like(compressed), c=data["give"])
    plt.show()
    return

# ------------------------- Fraud detection (шахрайство) -----------------------------------
def Fraud_detection (minimax_data):
    outliers_fraction = 0.1
    X = minimax_data.T
    clf = KNN(contamination=outliers_fraction)
    clf.fit(X)
    y_pred = clf.predict(X)
    # -------------------- Fraud detection (шахрайство) PLOT -------------------------------
    norm_data = StandardScaler().fit_transform(minimax_data.T)
    compressed = PCA(n_components=2).fit_transform(norm_data)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=compressed[:, 0], y=compressed[:, 1], hue=np.where(y_pred, "fraud", "no fraud"))
    plt.show()
    return

# -------------------- fraud, no fraud (повернення) PLOT -------------------------------
def fraud_no_fraud (minimax_data):
    outliers_fraction = 0.1
    X = minimax_data.T
    clf = KNN(contamination=outliers_fraction)
    clf.fit(X)
    y_pred = clf.predict(X)
    # ------------------------------ Binary scores -------------------------------------
    credit_given = minimax_data.T[data["give"] & ~y_pred].reset_index(drop=True)

    outliers_fraction = 0.026
    X = credit_given
    clf = KNN(contamination=outliers_fraction)
    clf.fit(X)
    y_pred = clf.predict(X)

    norm_data = StandardScaler().fit_transform(credit_given)
    compressed = PCA(n_components=2).fit_transform(norm_data)
    plt.figure(figsize=(10, 5))

    # -------------------- fraud, no fraud (повернення) PLOT -------------------------------
    sns.scatterplot(x=compressed[:, 0], y=compressed[:, 1], hue=np.where(y_pred, "will not return", "will return"))
    plt.show()
    return

# ----------------------------------------- main ---------------------------------------

if __name__ == '__main__':

    data = data_preparation("data_description.xlsx")
    print(data)
    minimax_info = pd.read_excel("d_segment_data_description_cleaning_minimax.xlsx")
    minimax_info = minimax_info[["Field_in_data", "Minimax"]]
    minimax_info = minimax_info.dropna()
    minimax_info.head()
    print(minimax_info.head())

    # ------------------------------------ minimax_data ---------------------------------
    minimax_data = data.T
    col_intersection = list(set(data.columns).intersection(minimax_info["Field_in_data"]))
    minimax_data = minimax_data.loc[col_intersection, :]
    minimax_data.head()

    criteria_count = len(minimax_data)
    criteria_values = minimax_data.astype(float)
    criteria_values = criteria_values.reset_index(drop=True)
    direction = (minimax_info.set_index("Field_in_data").loc[minimax_data.index, :]["Minimax"] == "max").values

    # -------------------------------------- integro - SCOR -----------------------------

    integro = min_max(voronin(criteria_values, np.ones(criteria_count) / criteria_count, direction))

    # ------------------------------------------- plot -----------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(np.sort(integro.clip(0, 0.2)), label="credit scores")
    plt.hlines(0.0185, 0, len(integro), color="r",
               label=f"threshold ({(integro <= 0.0185).sum() / len(integro):.3%})", )
    plt.legend()
    plt.grid()
    plt.show()

    # ------------------------------------------- scor_d_line -----------------------------------
    scor_d_line = data["give"] = integro <= 0.0185    # емпірічна константа відсікання рішення
    np.savetxt('Integro_Scor.txt', scor_d_line)       # файл інтегрованого показника - СКОРУ
    print('scor_d_line= ', scor_d_line)

    # ------------------------------ Linear discriminant analysis -------------------------------
    linear_discriminant_analysis(minimax_data)

    # ------------------------------- Fraud detection (шахрайство) ------------------------------
    Fraud_detection(minimax_data)

    # -------------------------------- fraud, no fraud (повернення) -----------------------------
    fraud_no_fraud(minimax_data)




