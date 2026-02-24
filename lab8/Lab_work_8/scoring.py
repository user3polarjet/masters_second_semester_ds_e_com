#----------------  скорінговий аналіз реальних даних мікрокредитування  ------------------

'''

Завдання:
1. Побудувати скорінгову оцінку
2. Проввести кластерізацію заявників

БАЗОВИЙ ПІДХІД - методи багатокритеріальної оптимізації - скорингу

Package                      Version
---------------------------- -----------
pip                          23.1
munch                        2.5.0
numpy                        1.23.5
pandas                       1.5.3
xlrd                         2.0.1
matplotlib                   3.6.2

'''

#--------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------- І. ПІДГОТОВКА ВХІДНИХ ДАНИХ ------------------------------
# 1.1. Парсінг файлу вхідних даних
#----------------------------------- парсинг файлу вхідних даних ------------------------------
d_sample_data = pd.read_excel('sample_data.xlsx', parse_dates=['birth_date'])
print('d_sample_data=', d_sample_data)                               # вивід усього DataFrame файлу sample_data.xls
Title_d_sample_data=d_sample_data.columns                            # імпорт імен стовпців DataFrame - d


# 1.2. Аналіз структури вхідних даних
#------------------------------- аналіз структури вхідних даних ------------------------------
print('-------------  назви стовпців DataFrame  -----------')
print(Title_d_sample_data)                                           # заголовок стовпців таблиці DataFrame
print('---------  типи даних стовпців DataFrame  -----------')
print(d_sample_data.dtypes)                                          # визначення типу даних таблиці DataFrame
print('---------  пропущені значення стовпців (суми)  ------')
print(d_sample_data.isnull().sum())                                  # визначення пропусків даних DataFrame


# 1.3. Парсінг файлу та аналіз структури скорингових індикаторів
#---------------------------- парсинг файлу пояснень параметрів ------------------------------
d_data_description = pd.read_excel('data_description.xlsx')
print('---------------  d_data_description  ---------------')
print('d_data_description=', d_data_description)                      # вивід усього DataFrame файлу data_description.xls
print('----------------------------------------------------')


# 1.4. Первинне формування скорингової таблиці
#-------------------------- сегментація ознак клієнта та кредиту  -------------------------------
d_segment_data_description_client_bank = d_data_description[(d_data_description.Place_of_definition == 'Вказує позичальник')
                                    | (d_data_description.Place_of_definition == 'параметри, повязані з виданим продуктом')]
n_client_bank = d_segment_data_description_client_bank['Place_of_definition'].size  # розмір стовпця
d_segment_data_description_client_bank.index = range(0, len(d_segment_data_description_client_bank))  # балансування індексів в сегменті 0-n
print('---------  d_segment_data_description_client_bank  -----------')
print('d_segment_data_description_client_bank=', d_segment_data_description_client_bank)
print('----------------------------------------------------')


# 1.5. Очищення даних
# 1.5.1. Аналіз перетину скорингових індикаторів та сегменту вхідних даних

#------ перевірка наявності індексів клієнта та кредиту з d_data_description в даних d_sample_data -------
print('-----------------------------------------')
b = d_segment_data_description_client_bank['Field_in_data']  # 'Field_in_data' - стовчик із назвами індиаторів скорінгу

if set(b).issubset(d_sample_data.columns):       # перевірка всього columns
    Flag_b = 'Flag_True'
else:
    Flag_b = 'Flag_False'
print('УВАГА! сегмент columns за співпадінням:', Flag_b)
#------ кількість співпадінь
n_columns = d_segment_data_description_client_bank['Field_in_data'].size
j=0
for i in range (0, n_columns):
    a = d_segment_data_description_client_bank['Field_in_data'][i]
    if set([a]).issubset(d_sample_data.columns):
        j=j+1                                   # кількість співпадінь для кожного columns
print('j=', j)
#------ індекси співпадінь
Columns_Flag_True = np.zeros((j))
j=0
for i in range (0, n_columns):
    a = d_segment_data_description_client_bank['Field_in_data'][i]
    if set([a]).issubset(d_sample_data.columns):  # перевірка кожного columns
        Flag = 'Flag_True'
        Columns_Flag_True[j]= i
        j=j+1
    else:
        Flag = 'Flag_False'
print('Індекси співпадінь', Columns_Flag_True)


# 1.5.2. Формування DataFrame даних з урахуванням відсутніх індикаторів скорингової таблиці
#------ DataFrame співпадінь
d_segment_data_description_client_bank_True = d_segment_data_description_client_bank.iloc[Columns_Flag_True]     # Формування
d_segment_data_description_client_bank_True.index = range(0, len(d_segment_data_description_client_bank_True))   # Балансування індиксів
print('------------ DataFrame співпадінь -------------')
print( d_segment_data_description_client_bank_True)
print('-----------------------------------------------')


# 1.5.3. Очищення скорингової таблиці від пропусків
#------- формування сегменту вхідних даних за рейтингом кліент + банк --------
b = d_segment_data_description_client_bank_True['Field_in_data']
d_segment_sample_data_client_bank = d_sample_data[b]
print('---- пропуски даних сегменту DataFrame --------')
print(d_segment_sample_data_client_bank.isnull().sum())                           # визначення пропусків даних DataFrame
print('-----------------------------------------------')


#------ вилучення строк та індикаторів з пропусками - СКОРИНГОВА КАРТА -------
# Очищення індикаторів скорингової таблиці
d_segment_data_description_cleaning = d_segment_data_description_client_bank_True.loc[
      (d_segment_data_description_client_bank_True['Field_in_data'] != 'fact_addr_start_date')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'position_id')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'employment_date')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'has_prior_employment')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'prior_employment_start_date')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'prior_employment_end_date')]
d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
      (d_segment_data_description_cleaning['Field_in_data'] != 'income_frequency_other')]
d_segment_data_description_cleaning.index = range(0, len(d_segment_data_description_cleaning))  # балансування індексів в сегменті 0-n
d_segment_data_description_cleaning.to_excel('d_segment_data_description_cleaning.xlsx')        # збереження очищених даних


# Очищення вхідних даних
d_segment_sample_cleaning = d_segment_sample_data_client_bank.drop( columns=['fact_addr_start_date', 'position_id', 'employment_date',
                                                                             'has_prior_employment', 'prior_employment_start_date',
                                                                             'prior_employment_end_date','income_frequency_other'])
d_segment_sample_cleaning.index = range(0, len(d_segment_sample_cleaning))                      # балансування індексів в сегменті 0-n
d_segment_sample_cleaning.to_excel('d_segment_sample_cleaning.xlsx')                            # збереження очищених даних
print('--- Контроль наявності пропусків даних після очищення на індикаторах ---')
print(d_segment_sample_cleaning.isnull().sum())                                                 # визначення пропусків даних DataFrame
print('---------- DataFrame вхідних даних - скорингова карта -----------')
print(d_segment_sample_cleaning)
print('----------------- DataFrame індикатори скорингу  ----------------')
print(d_segment_data_description_cleaning)
print('-----------------------------------------------------------------')


#----------------------------------- ІІ. ФОРМУВАННЯ СКОРИНГОВОЇ МОДЕЛІ ------------------------------

#--------------------------- Скорингова модель багатокритеріального оцінювання ----------------------


# 2.1. Парсінг файлу індикаторів (критеріїв) скорингової карти

d_data_description_minimax = pd.read_excel('d_segment_data_description_cleaning_minimax.xlsx') # інфологічна модель

# відбір критеріїв
d_segment_data_description_minimax = d_data_description_minimax.loc[
      (d_data_description_minimax['Minimax'] == 'min')
    | (d_data_description_minimax['Minimax'] == 'max')]
d_segment_data_description_minimax.index = range(0, len(d_segment_data_description_minimax))  # балансування індексів в сегменті 0-n
print('----------------- DataFrame d_segment_data_description_minimax  ----------------')
print(d_segment_data_description_minimax)
print('-----------------------------------------------------------------')
d_segment_data_description_minimax.to_excel('d_segment_data_description_minimax.xlsx')        # збереження очищених показників


# відбір даних за критеріями
d=d_segment_data_description_minimax['Field_in_data']
cols = d.values.tolist()                                                                      # перетворення стовпчика DataFrame у строку
d_segment_sample_minimax = d_segment_sample_cleaning[cols]
print('----------------- DataFrame d_segment_sample_minimax  ----------------')
print(cols)
print(d_segment_sample_minimax)
print('-----------------------------------------------------------------')
d_segment_sample_minimax.to_excel('d_segment_sample_minimax.xlsx')                            # збереження очищених даних


# 2.2. Парсінг файлу індикаторів (критеріїв) скорингової карти
# мінімальні, максимальні значення стовпчиків DataFrame
d_segment_sample_min = d_segment_sample_minimax[cols].min()
d_segment_sample_max = d_segment_sample_minimax[cols].max()
print('----------------- DataFrame: d_segment_sample_min  ----------------')
print(d_segment_sample_min)
print('----------------- DataFrame: d_segment_sample_max  ----------------')
print(d_segment_sample_max)


# 2.3.Нормування критеріїв
m = d_segment_sample_minimax['loan_amount'].size
n = d_segment_data_description_minimax['Field_in_data'].size
d_segment_sample_minimax_Normal=np.zeros((m, n))   # перехід в розрахунках до масиву numpy

delta_d=0.3                                        # коефіцієнт запасу при нормуванні
for j in range (0, len(d_segment_data_description_minimax)):
    columns_d = d_segment_data_description_minimax['Minimax'][j]
    if columns_d == 'min':
        columns_m=d_segment_data_description_minimax['Field_in_data'][j]
        for i in range(0, len(d_segment_sample_minimax)):
            max_max = d_segment_sample_max[j]+(2*delta_d)
            d_segment_sample_minimax_Normal[i, j] = (delta_d + d_segment_sample_minimax[columns_m][i]) / (max_max)
    else:
        for i in range(0, len(d_segment_sample_minimax)):
            min_min = d_segment_sample_max[j] + (2*delta_d)
            d_segment_sample_minimax_Normal[i, j] = (1 / (delta_d + d_segment_sample_minimax[columns_m][i]))/ (min_min)

print(d_segment_sample_minimax_Normal)
np.savetxt('d_segment_sample_minimax_Normal.txt', d_segment_sample_minimax_Normal)   # файл нормованих параметрів


# ---------------------------- структура нормованого масиву ----------------------------------------------
'''
d_segment_sample_minimax_Normal[i, j]:
m = d_segment_sample_minimax['loan_amount'].size              - кількість позичальників
n = d_segment_data_description_minimax['Field_in_data'].size  - кількість індикаторів скорингової таблиці
d_segment_sample_minimax_Normal=np.zeros((m, n))              - перехід в розрахунках до масиву numpy
'''

# --------------------------------------------------------------------------------------------------------

# 2.4.Інтегрована багатокритеріальна оцінка - SCOR
def Voronin (d_segment_sample_minimax_Normal, n):
    Integro = np.zeros((150))
    Scor = np.zeros((150))
    for i in range(0, 150):
         Sum_Voronin=0
         for j in range(0, n):
             Sum_Voronin = Sum_Voronin + ((1 - d_segment_sample_minimax_Normal[i,j]) ** (-1))
         Integro[i]=Sum_Voronin
         Scor[i] = 1000             # порог прийняття рішень про видачу кредиту
         np.savetxt('Integro_Scor.txt', Integro) # файл інтегрованого показника - СКОРУ
    plt.title('Integro_Scor')
    plt.plot(Integro)
    plt.plot(Scor)
    plt.show()
    return Integro

# -------------------------------- БЛОК ГОЛОВНИХ ВИКЛИКІВ ------------------------------

if __name__ == '__main__':

    Voronin(d_segment_sample_minimax_Normal, n)

