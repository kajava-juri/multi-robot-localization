import numpy as np

class EKFonSimData:
    def __init__(self, Pxy, Pr, Qxy, Qr, Rd, numRob):
        self.Pxy = Pxy
        self.Pr = Pr
        self.Qxy = Qxy
        self.Qr = Qr
        self.Rd = Rd
        self.numRob = numRob
        self.Pmatrix = np.zeros((3, 3, self.numRob, self.numRob))
        for i in range(self.numRob):
            for j in range(self.numRob):
                self.Pmatrix[0:2, 0:2, i, j] = np.eye(2)*Pxy
                self.Pmatrix[2, 2, i, j] = Pr

    # Replace the EKF method in EKFonSimData with this version
    def EKF(self, uNois, zNois, relativeState, ekfStride, reference_only=True, enforce_symmetry=True):
        """
        EKF update for simulated data.
        - uNois: 3 x numRob (vx, vy, yawRate)
        - zNois: numRob x numRob matrix of range observations (meters) or np.nan for missing
        - relativeState: 3 x numRob x numRob (x_ij, y_ij, yaw_ij) where state is j in i's body frame
        - ekfStride: integer
        - reference_only: if True, only update for i==0 (legacy behavior). If False, update for all i.
        - enforce_symmetry: if True, after updating X_ij set X_ji = inverse transform of X_ij
        """
        Q = np.diag([self.Qxy, self.Qxy, self.Qr, self.Qxy, self.Qxy, self.Qr])**2
        R = np.diag([self.Rd])**2  # observation covariance (1x1)
        dtEKF = ekfStride * 0.01
        n = self.numRob

        i_range = [0] if reference_only else list(range(n))

        for i in i_range:
            for j in range(n):
                if j == i:
                    continue
                z = zNois[i, j]
                # skip missing measurements
                if z is None or (isinstance(z, float) and np.isnan(z)):
                    continue
                # get controls
                uVix, uViy, uRi = uNois[:, i]
                uVjx, uVjy, uRj = uNois[:, j]
                xij, yij, yawij = relativeState[:, i, j]
                # state prediction
                dotXij = np.array([
                    np.cos(yawij)*uVjx - np.sin(yawij)*uVjy - uVix + uRi*yij,
                    np.sin(yawij)*uVjx + np.cos(yawij)*uVjy - uViy - uRi*xij,
                    uRj - uRi
                ])
                statPred = relativeState[:, i, j] + dotXij * dtEKF

                # jacobians
                jacoF = np.array([
                    [1,           uRi*dtEKF,  (-np.sin(yawij)*uVjx-np.cos(yawij)*uVjy)*dtEKF],
                    [-uRi*dtEKF,  1,          (np.cos(yawij)*uVjx-np.sin(yawij)*uVjy)*dtEKF ],
                    [0,           0,          1]
                ])
                jacoB = np.array([[-1,  0,  yij,  np.cos(yawij), -np.sin(yawij),  0],
                                [ 0, -1, -xij,  np.sin(yawij),  np.cos(yawij),  0],
                                [ 0,  0,   -1,                0,                0,  1]]) * dtEKF

                PPred = jacoF @ self.Pmatrix[:, :, i, j] @ jacoF.T + jacoB @ Q @ jacoB.T

                xij_p, yij_p, yawij_p = statPred
                dist = np.hypot(xij_p, yij_p)
                if dist < 1e-6:
                    # avoid division by zero; skip update if distance is effectively zero
                    continue

                zPred = dist
                jacoH = np.array([[xij_p/dist, yij_p/dist, 0.0]])
                resErr = z - zPred
                S = jacoH @ PPred @ jacoH.T + R
                # numerical guard: if S is near singular, skip update
                if S <= 1e-12:
                    continue
                K = PPred @ jacoH.T @ np.linalg.inv(S)
                # update state (ensure shapes)
                delta = (K @ np.array([[resErr]])).reshape((3,))
                newState = statPred + delta
                relativeState[:, i, j] = newState
                # update covariance
                self.Pmatrix[:, :, i, j] = (np.eye(3) - K @ jacoH) @ PPred

                # Optionally enforce simple symmetry: compute j's view from i's view
                if enforce_symmetry:
                    # X_ji is position of i in j's body frame; approximate by inverse transform
                    x_ij, y_ij, yaw_ij = relativeState[:, i, j]
                    # position of i in j frame is rotate(-yaw_ij) * (-[x_ij, y_ij])
                    c = np.cos(yaw_ij); s = np.sin(yaw_ij)
                    xji = -c * x_ij - s * y_ij
                    yji =  s * x_ij - c * y_ij
                    yawji = -yaw_ij
                    relativeState[:, j, i] = np.array([xji, yji, yawji])
                    # Keeping Pmatrix consistent is non-trivial; for now we copy the P with basic symmetry
                    self.Pmatrix[:, :, j, i] = self.Pmatrix[:, :, i, j].copy()

        return relativeState


class EKFonRealData:
# This is slight different from the above one, cause the logging data repeat during some steps.
# Therefore, only different sensor data triggers a new update.
    def __init__(self, Pxy, Pr, Qxy, Qr, Rd, numRob):
        self.Pxy = Pxy
        self.Pr = Pr
        self.Qxy = Qxy
        self.Qr = Qr
        self.Rd = Rd
        self.numRob = numRob
        self.Pmatrix = np.zeros((3, 3, self.numRob, self.numRob))
        for i in range(self.numRob):
            for j in range(self.numRob):
                self.Pmatrix[0:2, 0:2, i, j] = np.eye(2)*Pxy
                self.Pmatrix[2, 2, i, j] = Pr
        self.timeTick = np.zeros((self.numRob, self.numRob)) # to check difference
        self.zNoisOld = np.zeros((self.numRob, self.numRob)) # to check difference

    def EKF(self, uNois, zNois, relativeState):
    # Calculate relative position between robot i and j in i's body-frame
        Q = np.diag([self.Qxy, self.Qxy, self.Qr, self.Qxy, self.Qxy, self.Qr])**2
        R = np.diag([self.Rd])**2  # observation covariance
        uNois[0:2,:] = 0.88 * uNois[0:2,:] # a proportion gain, seems not important
        for i in range(1):
            for j in [jj for jj in range(self.numRob) if jj!=i]:
                if self.zNoisOld[i, j] != zNois[i, j]:
                    self.zNoisOld[i, j] = zNois[i, j]
                    dtEKF = 0.01*self.timeTick[i, j]
                    self.timeTick[i, j] = 1
                    # the relative state Xij = Xj - Xi
                    uVix, uViy, uRi = uNois[:, i]
                    uVjx, uVjy, uRj = uNois[:, j]
                    xij, yij, yawij = relativeState[:, i, j]
                    dotXij = np.array([np.cos(yawij)*uVjx - np.sin(yawij)*uVjy - uVix + uRi*yij,
                                    np.sin(yawij)*uVjx + np.cos(yawij)*uVjy - uViy - uRi*xij,
                                    uRj - uRi])
                    statPred = relativeState[:, i, j] + dotXij * dtEKF

                    jacoF = np.array([[1,       uRi*dtEKF,  (-np.sin(yawij)*uVjx-np.cos(yawij)*uVjy)*dtEKF],
                                    [-uRi*dtEKF, 1,         (np.cos(yawij)*uVjx-np.sin(yawij)*uVjy)*dtEKF ],
                                    [0,          0,         1]])
                    jacoB = np.array([[-1,  0,  yij,  np.cos(yawij), -np.sin(yawij),  0],
                                    [ 0, -1, -xij,  np.sin(yawij),  np.cos(yawij),  0],
                                    [ 0,  0,   -1,                0,                0,  1]])*dtEKF
                    PPred = jacoF@self.Pmatrix[:, :, i, j]@jacoF.T + jacoB@Q@jacoB.T
                    xij, yij, yawij = statPred
                    zPred = dist = np.sqrt(xij**2 + yij**2)+0.0001
                    jacoH = np.array([[xij/dist, yij/dist, 0]])
                    resErr = zNois[i, j] - zPred
                    S = jacoH@PPred@jacoH.T + R
                    K = PPred@jacoH.T@np.linalg.inv(S)
                    relativeState[:, [i], [j]] = statPred.reshape((3,1)) + K@np.array([[resErr]])
                    self.Pmatrix[:, :, i, j] = (np.eye(len(statPred)) - K@jacoH)@PPred
                else:
                    self.timeTick[i, j] += 1
        # print(np.trace(self.Pmatrix[:, :, i, j])) # for tuning the filter
        return relativeState