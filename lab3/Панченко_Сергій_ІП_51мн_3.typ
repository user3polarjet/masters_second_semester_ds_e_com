#set text(font: "Times New Roman", size: 12pt)
#set page(
  paper: "a4",
  margin: (top: 2cm, bottom: 2cm, left: 2.5cm, right: 2cm)
)
#align(center)[

*Міністерство освіти і науки України*

*Національний технічний університет України “Київський політехнічний інститут імені Ігоря Сікорського”*

*Кафедра обчислювальної техніки ФІОТ*
]

#v(1fr)

#align(center)[
  *ЗВІТ* \
  з лабораторної роботи №3 \
  з навчальної дисципліни «Технології Data Science для завдань електронної комерції»
]

#v(1em)

#align(left)[*Тема:*]
#align(center)[*"МАКЕТ ІНТЕЛЕКТУАЛЬНОЇ ERP СИСТЕМИ ПІДТРИМКИ ПРИЙНЯТТЯ РІШЕНЬ"*]

#v(2cm)

#align(right)[
  *Виконав:* \
  Студент 1 курсу кафедри ІПІ ФІОТ \
  Навчальної групи ІП-51мн \
  Сергій ПАНЧЕНКО \
  
  #v(1em)
  
  *Перевірив:* \
  Професор кафедри ОТ ФІОТ \
  Олексій ПИСАРЧУК
]

#v(1fr)

#align(center + bottom)[Київ 2026]

#pagebreak()

#set heading(numbering: (..nums) => nums.pos().map(str).join("."))
#show heading: it => {
  set text(weight: "bold", size: 12pt)
  set align(left)
  if it.level == 1 {
    counter(figure.where(kind: image)).update(0)
    it
  } else {
    it
  }
}
#show figure: it => {
  set align(center)
  it.body
  v(8pt, weak: true)
  it.supplement 
  [ ]
  context (it.counter.display(it.numbering))
  [ — ] 
  it.caption.body
}
#set figure(
  supplement: [Рисунок],
  numbering: (num) => {
    context {
      let h_num = counter(heading).at(here()).at(0)
      str(h_num) + "." + str(num)
    }
  }
)

#show ref: it => {
  let el = it.element
  if el != none and el.func() == figure {
    context {
      let num = el.counter.at(el.location())
      numbering(el.numbering, ..num)
    }
  } else {
    it
  }
}
#set par(leading: 0.65em, spacing: 0.65em, first-line-indent: (amount: 1.25cm, all: true), justify: true)
#set list(indent: 1.25cm)
#set enum(indent: 1.25cm)

= Мета
Виявити  дослідити  та  узагальнити  принципи  формалізації  задач,  синтезу математичних  моделей  для  автоматизації  процесів  підтримки  прийняття  рішень  в інтелектуальних  ERP системах: програмування обмежень  –  CP-SAT;  багатокритеріальні задачі – Multicriteria decision analysis.

= Завдання

*І рівень складності:*
Розробити програмний скрипт, що реалізує оцінювання ефективності вибору житла. Вихідні дані містять 12 критеріїв, з яких 4 – максимізованих, решта – мінімізованих. Кількість аналогічних товарів (альтернатив) – 8. Вхідні дані занесені у файл. 

= Результати виконання лабораторної роботи (Рівень І)

== Синтезована математична модель

Для розв'язання задачі вибору житла застосовано метод багатокритеріального аналізу рішень (MCDA) на основі зваженої суми з попередньою мінімаксною нормалізацією критеріїв.

1. *Нормалізація максимізованих критеріїв* (площа, кількість кімнат, поверх, кількість балконів). Чим більше значення, тим краще:
$ x_{"norm"} = (x - x_{"min"}) / (x_{"max"} - x_{"min"}) $

2. *Нормалізація мінімізованих критеріїв* (ціна, відстань до метро/центру/супермаркету, рівень злочинності, вік будинку, комунальні послуги, рівень шуму). Чим менше значення, тим краще:
$ x_{"norm"} = (x_{"max"} - x) / (x_{"max"} - x_{"min"}) $

3. *Обчислення інтегральної оцінки* (Integral Score). Використовується адитивна згортка з урахуванням вектора вагових коефіцієнтів $W$:
$ I = sum_{i=1}^{n} w_i \cdot x_{"norm"}^{(i)} $
де $sum_{i=1}^{n} w_i = 1$. Альтернатива з максимальним значенням $I$ вважається оптимальною.

== Блок-схема алгоритму та її опис

#import "@preview/fletcher:0.5.5" as fletcher: diagram, node, edge
#import fletcher.shapes: diamond, rect, pill, parallelogram

#align(center)[
  #let w = 7.5cm
  #scale(x: 80%, y: 80%, reflow: true)[
      #diagram(
      node-stroke: 1pt, edge-stroke: 1pt, node-inset: 8pt,
      
      node((0,0), box(width: w, align(center)[Початок]), shape: pill, corner-radius: 10pt),
      edge("-|>"),
      
      node((0,1), box(width: w, align(center)[Перевірка наявності файлу\n`housing_data.csv` та зчитування даних]), shape: parallelogram),
      edge("-|>"),
      
      node((0,2), box(width: w, align(center)[Розподіл критеріїв на\nмаксимізовані (4) та мінімізовані (8)]), shape: rect, corner-radius: 2pt),
      edge("-|>"),
      
      node((0,3), box(width: w, align(center)[Мінімаксна нормалізація\nвсіх критеріїв до діапазону [0; 1]]), shape: rect, corner-radius: 2pt),
      edge("-|>"),

      node((0,4), box(width: w, align(center)[Розрахунок інтегральної оцінки (MCDA)\nчерез матричне множення на ваги]), shape: rect, corner-radius: 2pt),
      edge("-|>"),

      node((0,5), box(width: w, align(center)[Сортування альтернатив\nза спаданням інтегральної оцінки]), shape: rect, corner-radius: 2pt),
      edge("-|>"),

      node((0,6), box(width: w, align(center)[Вивід результатів у консоль та\nзбереження графіка у SVG]), shape: parallelogram),
      edge("-|>"),

      node((0,7), box(width: w, align(center)[Кінець]), shape: pill, corner-radius: 10pt)
    )
  ]
]

Алгоритм розпочинає роботу з ініціалізації датасету (генерує його, якщо файл відсутній). Після завантаження у `pandas.DataFrame`, масив даних розділяється на дві групи за типом оптимізації (max/min). Застосовується відповідна формула нормалізації, що приводить всі значення до єдиної розмірності. Далі обчислюється скалярний добуток матриці нормалізованих значень на вектор ваг, формуючи фінальний рейтинг, який візуалізується.

== Опис структури проекту програми

- `level1.py` — головний скрипт для виконання багатокритеріального аналізу.
- `build/` — директорія для збереження результатів.
  - `housing_data.csv` — згенерований датасет із 8 альтернативами та 12 критеріями.
  - `housing_ranking.svg` — векторний графік рейтингу альтернатив.

== Результати роботи програми

Система успішно проаналізувала 8 варіантів житла. Результати ранжування (від найкращого до найгіршого):
- House_B: 0.6733
- House_F: 0.6529
- House_G: 0.5801
- House_D: 0.5553
- House_C: 0.5509
- House_A: 0.5394
- House_E: 0.5000
- House_H: 0.4967

#figure(
  image("build/housing_ranking.svg", width: 100%), 
  caption: [Рейтинг варіантів житла за методом MCDA]
) <mcda_results>

== Програмний код

#let embed_python(file_path) = {
  heading(file_path, level: 3)
  raw(read(file_path), lang: "python", block: true)
}

#embed_python("level1.py")

== Аналіз результатів відлагодження

Графічні (Рис. @mcda_results) та числові результати підтверджують правильність роботи математичної моделі. `House_B` став беззаперечним лідером рейтингу (оцінка 0.6733), оскільки він має найвищі показники за найбільш вагомими критеріями: низьку ціну (вага 0.20), мінімальну відстань до метро (вага 0.15) та низький рівень злочинності. Водночас `House_E`, незважаючи на найбільшу площу та кількість кімнат, опинився на передостанньому місці через надмірно високу вартість та віддаленість від інфраструктури, що були суттєво "оштрафовані" мінімізованою нормалізацією.

= Результати виконання лабораторної роботи (Рівень ІІ)

== Синтезована математична модель та опис датасету

Для розв'язання задачі багатокритеріального оцінювання маршрутів переміщення від дому (Харківське шосе) до місця навчання (КПІ) було зібрано та проаналізовано дані з 11 реальних маршрутів за допомогою сервісу Google Maps (експорт у PDF з подальшим перетворенням у графічні файли та структуруванням у CSV-датасет). 

Оцінювання проводилося за 7 критеріями, які були розділені на дві групи:
1. *Мінімізовані критерії:* `Total_Time_min` (загальний час, вага 0.30), `Walking_Time_min` (час пішки, вага 0.10), `Transfers_count` (кількість пересадок, вага 0.05), `Cost_UAH` (вартість, вага 0.20), `Traffic_Jam_Risk` (ризик заторів, вага 0.10).
2. *Максимізовані критерії:* `Comfort_Level` (комфорт, вага 0.15), `Reliability` (надійність розкладу, вага 0.10).

Математична модель використовує мінімаксну нормалізацію:
- Для максимізованих: $x_{"norm"} = (x - x_{"min"}) / (x_{"max"} - x_{"min"})$
- Для мінімізованих: $x_{"norm"} = (x_{"max"} - x) / (x_{"max"} - x_{"min"})$

Інтегральна ефективність кожного маршруту розраховується як зважена сума:
$ I = sum_{j=1}^{7} w_j \cdot x_{"norm"}^{(j)} $

== Картографічні дані маршрутів (Вихідні дані)

Нижче наведено візуалізацію відібраних альтернатив для переміщення автомобілем, громадським транспортом та пішки.

#figure(image("car_0.png", width: 80%), caption: [Автомобіль №1 (через центр)]),
#figure(image("car_1.png", width: 80%), caption: [Автомобіль №2 (Набережне шосе)]),
#figure(image("car_2.png", width: 80%), caption: [Автомобіль №3 (Південний міст)]),
#figure(image("transit_0.png", width: 80%), caption: [Транзит №1 (Авт. 316 + М1)]),
#figure(image("transit_1.png", width: 80%), caption: [Транзит №2 (Авт. 511 + М1)]),
#figure(image("transit_2.png", width: 80%), caption: [Транзит №3 (Авт. 45 + М1)]),
#figure(image("transit_3.png", width: 80%), caption: [Транзит №4 (Електричка + Трамвай)]),
#figure(image("transit_4.png", width: 80%), caption: [Транзит №5 (Міська електр. + Трамвай)]),
#figure(image("transit_5.png", width: 80%), caption: [Транзит №6 (Трамвай 27 + М1)]),
#figure(image("walk_0.png", width: 80%), caption: [Пішки №1 (через центр)]),
#figure(image("walk_1.png", width: 80%), caption: [Пішки №2 (через Поділ)])

#v(1em)

#show figure.where(kind: table): it => {
  set par(first-line-indent: (amount: 0cm, all: true))
  set align(left)
  it.supplement 
  [ ]
  context (it.counter.display(it.numbering))
  [ — ]
  it.caption.body
  v(8pt, weak: true)
  it.body
}

#let vertical_header(content) = {
  rotate(-90deg, reflow: true, [*#content*])
}

#let route-data = csv("routes_data.csv")
#let header = route-data.at(0)
#let body_rows = route-data.slice(1)

#figure(
  table(
    columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: center + horizon,
    fill: (col, row) => if row == 0 { luma(240) } else { none },
    ..header.enumerate().map(((i, cell)) => {
      if i < 2 { [*#cell*] } 
      else { vertical_header(cell) }
    }),
    ..body_rows.flatten()
  ),
  kind: table,
  supplement: [Таблиця],
  caption: [Зведені дані маршрутів]
) <routes_table_data>

== Блок-схема алгоритму та її опис

#align(center)[
  #let w = 7.5cm
  #scale(x: 80%, y: 80%, reflow: true)[
      #diagram(
      node-stroke: 1pt, edge-stroke: 1pt, node-inset: 8pt,
      
      node((0,0), box(width: w, align(center)[Початок]), shape: pill, corner-radius: 10pt),
      edge("-|>"),
      
      node((0,1), box(width: w, align(center)[Зчитування даних маршрутів з `routes_data.csv`]), shape: parallelogram),
      edge("-|>"),
      
      node((0,2), box(width: w, align(center)[Мінімаксна нормалізація 7 критеріїв\nзведення до діапазону [0; 1]]), shape: rect, corner-radius: 2pt),
      edge("-|>"),

      node((0,3), box(width: w, align(center)[Адитивна згортка (MCDA)\nна основі вагових коефіцієнтів студентських пріоритетів]), shape: rect, corner-radius: 2pt),
      edge("-|>"),

      node((0,4), box(width: w, align(center)[Сортування маршрутів за спаданням\nінтегральної оцінки ефективності]), shape: rect, corner-radius: 2pt),
      edge("-|>"),

      node((0,5), box(width: w, align(center)[Генерація та збереження\n кольорової гістограми у SVG]), shape: parallelogram),
      edge("-|>"),

      node((0,6), box(width: w, align(center)[Кінець]), shape: pill, corner-radius: 10pt)
    )
  ]
]

== Результати роботи програми (Рівень ІІ)

Система успішно проаналізувала 11 маршрутів. Результати ранжування:
1. `Transit_CityTrain_Tram` (Транзит): 0.6790
2. `Transit_Bus45_M1` (Транзит): 0.6643
3. `Transit_Bus316_M1` (Транзит): 0.6552
4. `Transit_Tram27_M1` (Транзит): 0.6520
5. `Transit_Train_Tram` (Транзит): 0.6496
6. `Car_Center` (Авто): 0.6437
7. `Car_Pivdennyi` (Авто): 0.6431
8. `Car_Naberezhne` (Авто): 0.6303
9. `Transit_Bus511_M1` (Транзит): 0.5748
10. `Walk_Center` (Пішки): 0.4697
11. `Walk_Myru` (Пішки): 0.4500

#figure(
  image("build/routes_ranking.svg", width: 100%), 
  caption: [Рейтинг маршрутів від дому до КПІ за методом MCDA]
) <routes_results>

== Програмний код (Рівень ІІ)

#embed_python("level2.py")

== Аналіз результатів відлагодження

Результати роботи алгоритму (Рис. @routes_results) демонструють високу адекватність розробленої моделі реальним умовам. 

Найвищу оцінку (0.6790) отримав маршрут `Transit_CityTrain_Tram` (Міська електричка + Трамвай). Незважаючи на наявність пересадок, цей маршрут забезпечує ідеальний баланс: він дешевий (23 грн) та має найвищу надійність (мінімальний ризик заторів), оскільки рейковий транспорт рухається виділеними лініями.

Автомобільні маршрути (`Car_Center`, `Car_Pivdennyi`) отримали середні оцінки (близько 0.64). Хоча вони забезпечують найвищий рівень комфорту (10/10) та найменший час у дорозі (35-40 хв), модель суттєво "оштрафувала" їх за високу вартість поїздки (витрати на пальне понад 70-90 грн) та високий ризик потрапити у затор у години пік.

Пішохідні маршрути очікувано опинилися на останніх місцях (0.45-0.46), оскільки час у дорозі (понад 4 години) та фізична втома роблять їх абсолютно неефективними для щоденних поїздок на навчання, незважаючи на нульову фінансову вартість.