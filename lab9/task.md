Технології Data Science для завдань електронної комерції 
 
Лабораторна робота No_9  
МАКЕТ GIS СИСТЕМИ 
 
Мета роботи: 
дослідити  виявити  та  узагальнити  особливості  впровадження  геоінформаційних 
технологій в процесах Data Science. 

І. SKILLS, які прокачуємо. 
1. Архітектурне проектування програмних GIS систем. 
2. Практика пошуку, парсингу обробки та візуалізації геоданих – цифрових векторних 
карт формату Shapefile. 
3.  Практика  роботи  із  цифровими векторними  картами: завантаження,  візуалізація 
точкових та полігонних об’єктів, фільтрація даних, створення цифрових векторних карт, 
робота з атрибутами просторових об’єктів та їх аналітика. 
4.  Аналітика  геоданих,  візуалізація  результатів  аналітики  даних  з  використанням 
цифрових векторних карт. 
5.  R&D  процеси застосування технологій машинного навчання (Machine  Learning): 
Statistical Learning; k-Means / k-середніх; «найближчого  сусіда»  /  k-nearest neighbors 
algorithm; опорних векторів (support vector machine - SVM); багатокритеріальне оцінювання 
/ multi-criteria scoring. 
6. Прикладне використання бібліотек: Geopandas, Pandas, GDAL, Numpy, Matplotlib. 
7. Візуалізація та аналіз результатів розрахунків. 
8. Практика верифікація багаторівневих програмних систем.

ІІІ. Завдання. 
В  інтересах  потенційних  замовників  –  державних  та  комерційних  установ  R&D 
лабораторія  провідної  ІТ-компанії  розробляє  макет  GIS  системи.  Необхідно  розробити 
програмний  скрипт  Backend  – компоненту  GIS системи  з  функціоналом,  за  рівнями 
складності.

І рівень складності 7 балів. 
Реалізувати розрахунки та побудову / візуалізацію на цифровій векторній карті сітку 
відстаней між пожежними станціями США (див. приклад Лекцій No16). Кількість пожежних 
станцій для побудови сітки відстаней обрати самостійно. Розрахувати середню відстань між 
пожежними  станціями.  Провести  верифікацію  результатів  розрахунку  відстаней  за 
одиницями виміру.

***  додатково  +  2  бали:  визначити  щільність  розташування  пожежних  станцій 
територією США; встановити та відобразити центроїди районів з найбільшою щільністю; 
відобразити межи п’яти районів з найбільшою щільністю на цифровій карті. 
Допускається самостійне обрання точкових об’єктів будь-якого типу для розрахунку 
відстаней  та  їх  візуалізації  на  цифровій  векторній  карті  (із  самостійним  пошуком 
відповідних прошарків цифрової векторної карти). 


lab9/Lab_work_9_example/GIS_distance_example/
├── example_1.py
├── Fire_Stations
│   ├── Fire_Stations.cpg
│   ├── Fire_Stations.dbf
│   ├── Fire_Stations.prj
│   ├── Fire_Stations.shp
│   ├── Fire_Stations.shx
│   └── Fire_Stations.xml
└── Fire_Stations_SELECTION
    ├── Fire_St_SELECT.cpg
    ├── Fire_St_SELECT.dbf
    ├── Fire_St_SELECT.prj
    ├── Fire_St_SELECT.shp
    └── Fire_St_SELECT.shx

3 directories, 12 files
(venv) user@archlinux:~/university/masters_second_semester_ds_e_com/lab9$ 



#-------------------- Приклади можливостей та використання методів geopandas ---------------------------

'''
Розрахунок відмтані між двома пожежними станціями за даними від:
https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::fire-stations/explore

Цей набір даних містить точкові характеристики, що представляють розташування будівель пожежних станцій у
Сполучених Штатах, окрузі Колумбія, Пуерто-Ріко та Американських Віргінських островах.
Метою цієї колекції є зображення місць розташування пожежних станцій на картографічних виробах загального призначення.
'''

import pathlib
import os
import geopandas as gpd
import matplotlib.pyplot as plt

SCRIPT_PATH = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR = SCRIPT_PATH.parent
BUILD_DIR = SCRIPT_DIR / 'build'

#------------ парсінг фашлу карти *.shp - формату -------------
filename = SCRIPT_DIR / "Lab_work_9_example" / "GIS_distance_example" / "Fire_Stations" / "Fire_Stations.shp"
fire_stations = gpd.read_file(filename)
print(type(fire_stations), 'Карта формата *.shp')
print(fire_stations)

# візуалізація
fire_stations.plot()
plt.show()

distance = fire_stations.iloc[0].geometry.distance(fire_stations.iloc[2].geometry)
print(distance)

fire_stations = fire_stations.to_crs('EPSG:4326')
distance = fire_stations.iloc[0].geometry.distance(fire_stations.iloc[2].geometry) / 1000
print(distance)


#-------------------- система координат -----------------------
'''
EPSG:4326 WGS 84 -- WGS84 - Всесвітня геодезична система координат 1984р., використовується в GPS - навігації
'''
print('Система координат', fire_stations.crs)

#-------------------- створення Shapefile -----------------------

filename_out = SCRIPT_DIR / "Lab_work_9_example" / "GIS_distance_example" / "Fire_Stations_SELECTION" / "Fire_St_SELECT.shp"

# вибір 2 рядків
selection = fire_stations[0:2]

# запис відібраних радків в новий Shapefile
selection.to_file(filename_out)

# парсінг фашлу карти *.shp - формату
fire_stations_out = gpd.read_file(filename_out)

# візуалізація
fig, ax = plt.subplots(figsize=(8, 4))
fire_stations_out.plot(ax=ax, alpha=0.4, color="grey", zorder=1)
fire_stations.plot(ax=ax, markersize=20, color="blue", marker="o", zorder=2)
fire_stations_out.plot(ax=ax, markersize=20, color="red", marker="o", zorder=2)
plt.show()
