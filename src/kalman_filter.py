from dataclasses import dataclass
import numpy as np


@dataclass
class KalmanModel:

    # Discrete-Time Index
    n: float

    # Размерности входных значений
    nx: int
    nz: int
    nu: int

    # State Transition Matrix
    F: np.typing.NDArray[np.float64]  # A: nx x nx

    # Control Matrix
    G: np.typing.NDArray[np.float64]  # B: nx x nu

    # Observation Matrix
    H: np.typing.NDArray[np.float64]  # CT: nz x nz

    # === Noizes ===

    # Process Noize
    w: np.typing.NDArray[np.float64]  # w: nx x 1

    # Measurement Noize
    v: np.typing.NDArray[np.float64]  # v: nz x 1

    # ======

    # Measurements
    z: np.typing.NDArray[np.float64]  # z_k:  nz x 1

    # Input
    u: np.typing.NDArray[np.float64]  # u_k:  nu x 1


class KalmanFilter:

    @staticmethod
    def filter(kalman_model: KalmanModel, x0, P0: np.typing.NDArray[np.float64]):
        F = kalman_model.F
        G = kalman_model.G
        v = kalman_model.v
        w = kalman_model.w
        R = v @ v.T
        Q = w @ w.T
        N = len(kalman_model.z)
        H = kalman_model.H
        I = np.eye(kalman_model.nx)

        u = kalman_model.u
        z = kalman_model.z

        # === Estimates ===

        # Estimate State Array
        xe = np.empty((N, kalman_model.nx, 1))  # xe_k: nx x 1
        xe[0] = x0

        # Estimate Covariance Array
        Pe = np.empty((N, kalman_model.nx, kalman_model.nx))  # Pe_k: nx x nx
        Pe[0] = P0

        # Result with real value estimate
        ze = np.empty((N, kalman_model.nz, 1))

        # === Predicts ===

        # Predict State Array
        xp = np.empty((N, kalman_model.nx, 1))  # xp_k: nx x 1

        # Predict Covariance Array
        Pp = np.empty((N, kalman_model.nx, kalman_model.nx))  # Pp_k: nx x nx

        for i in range(1, N):
            # Predict
            xp[i] = F @ xe[i - 1] + G @ u[i - 1]
            Pp[i] = F @ Pe[i - 1] @ F.T + Q
            # Update
            K = Pp[i] @ H.T @ np.linalg.inv(H @ Pp[i] @ H.T + R)
            xe[i] = xp[i] + K @ (z[i] - H @ xp[i])
            Pe[i] = (I - K @ H) @ Pp[i] @ (I - K @ H).T + K @ R @ K.T
            # Convert to result
            ze[i] = H @ xe[i]

        return ze
