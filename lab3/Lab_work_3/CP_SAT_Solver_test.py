#----------------------------- Decision Support System (DSS) -------------------------------------

'''

Демонстраційний приклад розвя'зку задачі пошуку оптимального рішення
лінійної оптимізаційної задачі в цілих числах:
- функція ефективності одна;
- множина вимог ефективності враховується в обмеженнях;
- цільова функція та обмеження - лінійни;
- оптимізатор - з бібліотеки Google OR-Tools.

Першоджерело:
https://developers.google.com/optimization/cp/integer_opt_cp

'''

from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import numpy as np


def cp_model_solver():
    # Оптимізаційна математична модель
    model = cp_model.CpModel()
    var_upper_bound = max(50, 45, 37)
    x = model.NewIntVar(0, var_upper_bound, 'x')
    y = model.NewIntVar(0, var_upper_bound, 'y')
    z = model.NewIntVar(0, var_upper_bound, 'z')

    # Обмеження
    model.Add(2 * x + 7 * y + 3 * z <= 50)
    model.Add(3 * x - 5 * y + 7 * z <= 45)
    model.Add(5 * x + 2 * y - 6 * z <= 37)

    # Цільова функція ефективності
    efficiency_function = 2 * x + 2 * y + 3 * z

    model.Maximize(efficiency_function) # Максимізоція цільової функції
    # model.Minimize(efficiency_function)  # Мінімізація цільової функції

    # Вирішувач
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print()
        print('Maximum of objective function: %i' % solver.ObjectiveValue())
        xout = solver.Value(x)
        yout = solver.Value(y)
        zout = solver.Value(z)
        print('x= ', xout)
        print('y= ', yout)
        print('z= ', zout)

        print('---------------Оптимальная-эффективность---------------------')
        MaximusOPT = 2 * xout + 2 * yout + 3 * zout
        print('MaximusOPT= ', MaximusOPT)

        # -------------- Аналіз отриманого рішення ---------------------------
        '''
        ---------------Максимум-Максиморум для заданих обмежень ------------
        MaxMaxX:  2x+7y+3z=50, y=0,z=0, xmax=25    (1 нерівність)
        MaxMaxY:  5x+2y-6z=37, x=0,z=0, ymax=18.5  (3 неравекство)
        MaxMaxZ:  2x+7y+3z=50, x=0,y=0, zmax=16.5  (1 неравекство)    
        '''
        print('---------------- Максимум-Максиморум ------------------------')
        xmax = 25
        ymax = 18.5
        zmax = 16.5
        MaximusP = 2 * xmax + 2 * ymax + 3 * zmax
        print('MaximusP= ', MaximusP)

        '''
        --------------- Мінімум-Мініморум для заданих ообмежень -----------
        MinMinX:  5x+2y-6z=37, y=0,z=0, xmin=7.4    (1 нерівність)
        MinMinY:  3x-5y+7z=45, x=0,z=0, ymin=-9     (3 неравекство)
        MinMinZ:  5x+2y-6z=37, x=0,y=0, zmin=-6.16  (1 неравекство)    
        '''
        xmin = 7.4
        ymin = -9
        zmin = -6.16
        print('---------------- Мінімум-Мініморум ------------------------')
        MinimusM = 2 * xmin + 2 * ymin + 3 * zmin
        print(' MinimusM= ', MinimusM)

    return


def plot_solver():
    # -------- Графічна інтерпретація рішення "точкове" ----------
    fig = plt.figure()
    plt3d = fig.add_subplot(projection='3d')
    # ------------------ оптималье рішення-------------------------
    plt3d.scatter(7, 3, 5, color='green')
    # ------------------ максимальне рішення-----------------------
    plt3d.scatter(25, 18.5, 16.5, color='blue')
    # ------------------ мінімальне рішення -----------------------
    plt3d.scatter(7.4, -9, -6.16, color='red')
    plt.show()

    # -------- Графічна інтерпретація рішення "площинне" ----------
    x1, y1 = np.meshgrid(range(-10, 30), range(-10, 30))
    z1 = (2 * x1 + 2 * y1 + 0) * (-1.0 / 3)
    z2 = (2 * x1 + 7 * y1 - 50) * (-3.0 / 3)
    z3 = (3 * x1 - 5 * y1 - 45) * (-7.0 / 3)
    z4 = (5 * x1 + 2 * y1 - 37) * (+6.0 / 3)
    fig = plt.figure()
    plt3d = fig.add_subplot(projection='3d')
    plt3d.plot_surface(x1, y1, z1, alpha=0.5)
    plt3d.plot_surface(x1, y1, z2, alpha=0.7)
    plt3d.plot_surface(x1, y1, z3, alpha=0.8)
    plt3d.plot_surface(x1, y1, z4, alpha=0.9)
    # ------------------ оптималье рішення-------------------------
    plt3d.scatter(7, 3, 5, color='green')
    # ------------------ максимальне рішення-----------------------
    plt3d.scatter(25, 18.5, 16.5, color='blue')
    # ------------------ мінімальне рішення -----------------------
    plt3d.scatter(7.4, -9, -6.16, color='red')
    plt.show()

    return

# ----------------------- головні виклики -------------------------
if __name__ == '__main__':
   cp_model_solver()
   plot_solver()



