from typing import Callable
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from kalman_filter import KalmanFilter, KalmanModel


class SpringWeightModel(KalmanModel):
    n = 0.1
    nx = 2
    nz = 1
    nu = 1
    F = np.array([[0, 1], [-1, 2 - n**2 * (1 / 3) ** 2]])
    G = np.array([[0], [n**2]])
    H = np.array([[1, 0]])
    w = np.array([[0]])
    v = np.array([[5]])

    def __init__(self, size: int, f: Callable):
        F = self.F
        G = self.G
        H = self.H
        w = self.w
        v = self.v

        N = int(size / self.n)
        abscissa = np.linspace(0, size, N)
        u = np.array([[[f(x)]] for x in abscissa])

        x = np.zeros((N, self.nx, 1))

        z = np.empty((N, self.nz, 1))
        z_clean = np.empty((N, self.nz, 1))
        for i in range(1, N):
            x[i] = F @ x[i - 1] + G @ u[i - 1] + w
            z_clean[i] = H @ x[i]
            z[i] = H @ x[i] + np.random.normal(0, v)

        self.x = x
        self.z = z
        self.u = u
        self.abscissa = abscissa
        self.z_clean = z_clean


class F16Model(KalmanModel):
    n = 1
    nx = 2
    nz = 2
    nu = 1
    F = np.array([[-0.6, 0.95], [-4.3, -1.2]])
    G = np.array([[-0.115], [-5.157]])
    H = np.array([[1, 0], [0, 1]])
    w = np.array([[0]])
    v = np.array([[5]])

    def __init__(self, size: int, f: Callable):
        F = self.F
        G = self.G
        H = self.H
        w = self.w
        v = self.v

        N = int(size / self.n)
        abscissa = np.linspace(0, size, N)
        u = np.array([[[f(x)]] for x in abscissa])

        x = np.zeros((N, self.nx, 1))

        z = np.empty((N, self.nz, 1))
        z_clean = np.empty((N, self.nz, 1))
        for i in range(1, N):
            x[i] = F @ x[i - 1] + G @ u[i - 1] + w
            z_clean[i] = H @ x[i]
            z[i] = H @ x[i] + np.random.normal(0, v)

        self.x = x
        self.z = z
        self.u = u
        self.abscissa = abscissa
        self.z_clean = z_clean


if __name__ == "__main__":
    spring_weight_model = SpringWeightModel(400, lambda x: np.sign(np.sin(0.05 * x)))
    prediction = KalmanFilter.filter(
        spring_weight_model, np.array([[0], [0]]), np.eye(spring_weight_model.nx) * 30
    )

    prediction = prediction.reshape((len(spring_weight_model.abscissa)))
    z_clean = spring_weight_model.z_clean.reshape((len(spring_weight_model.abscissa)))

    matplotlib.use("MacOSX")

    # plt.plot(
    #     spring_weight_model.abscissa,
    #     spring_weight_model.u.reshape((len(spring_weight_model.abscissa))),
    #     color="red",
    #     label="внешняя сила, действующая на груз",
    # )
    plt.plot(
        spring_weight_model.abscissa,
        spring_weight_model.z.reshape((len(spring_weight_model.abscissa))),
        color="green",
        label="зашумлённое движение груза, измерения",
        linewidth=0.2,
    )
    plt.plot(
        spring_weight_model.abscissa,
        z_clean,
        color="blue",
        label="реальное движение груза",
    )
    plt.plot(
        spring_weight_model.abscissa, prediction, color="red", label="оценка значения"
    )
    plt.plot(
        spring_weight_model.abscissa,
        prediction - z_clean,
        color="orange",
        label="погрешность",
    )
    plt.legend()
    plt.show()
