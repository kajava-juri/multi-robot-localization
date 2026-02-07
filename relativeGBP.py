import numpy as np

class Variable:
    def __init__(self, name, dim=3):
        self.name = name
        self.dim = dim
        self.id = int(name[4:])  # Extract id from 'poseX'
        self.incoming_messages = {}   # factor_name -> (eta, Lambda)

    def get_belief(self):
        eta_total = np.zeros(self.dim)
        Lambda_total = np.zeros((self.dim, self.dim))
        for eta, Lambda in self.incoming_messages.values():
            eta_total += eta
            Lambda_total += Lambda
        if np.linalg.det(Lambda_total) < 1e-12:
            return np.zeros(self.dim), np.eye(self.dim) * 1e-6
        Sigma = np.linalg.inv(Lambda_total)
        mu = Sigma @ eta_total
        return mu, Sigma

    def get_msg_to_factor(self, factor_name):
        eta = np.zeros(self.dim)
        Lambda = np.zeros((self.dim, self.dim))
        for f_name, (e, L) in self.incoming_messages.items():
            if f_name != factor_name:
                eta += e
                Lambda += L
        if np.linalg.det(Lambda) < 1e-12:
            return np.zeros(self.dim), np.zeros((self.dim, self.dim))
        return eta, Lambda

class GBPonSimData:
    def __init__(self, Pxy=10.0, Pr=0.1, Qxy=0.25, Qr=0.4, Rd=0.1, numRob=4):
        self.numRob = numRob
        self.Qxy = Qxy      # std dev [m] for motion noise in x/y
        self.Qr = Qr        # std dev [rad] for yaw process noise
        self.Rd = Rd        # std dev [m] for range observation noise

        self.max_iter = 40
        self.beta = 0.35           # damping (lower to stabilize)
        self.anchor_strength = 1e6

        self.variables = {i: Variable(f'pose{i}') for i in range(numRob)}
        # store last absolute poses (estimates) as 3 x N
        self.abs_poses = np.zeros((3, numRob))

        # initial covariance (not heavily used)
        self.Pmatrix = np.zeros((numRob, 3, 3))
        for i in range(numRob):
            self.Pmatrix[i, 0:2, 0:2] = np.eye(2) * Pxy
            self.Pmatrix[i, 2, 2] = Pr

        print(self.Pmatrix)
        breakpoint

    def GBP(self, uNois, zNois, relativeState, ekfStride):
        dt = ekfStride * 0.01
        n = self.numRob

        # 1) PREDICT step: propagate abs_poses -> abs_pred using motion model
        abs_pred = np.copy(self.abs_poses)
        for i in range(n):
            vx, vy, yawrate = uNois[:, i]
            yaw = abs_pred[2, i]
            c, s = np.cos(yaw), np.sin(yaw)
            # rotate body-frame velocity into world frame
            abs_pred[0, i] += dt * (c * vx - s * vy)
            abs_pred[1, i] += dt * (s * vx + c * vy)
            abs_pred[2, i] += dt * yawrate

        # current_mu map used for linearization in factor messages
        current_mu = {i: abs_pred[:, i].copy() for i in range(n)}

        # 2) Reset messages (keep only anchor on robot 0)
        for v in self.variables.values():
            v.incoming_messages.clear()

        # anchor robot 0 to its predicted pose (not origin, since robot 0 moves!)
        Lambda0 = np.eye(3) * self.anchor_strength
        eta0 = Lambda0 @ abs_pred[:, 0]  # anchor at predicted position of robot 0
        self.variables[0].incoming_messages['anchor'] = (eta0, Lambda0)

        # 3) Build factors using the correct linearization point (use abs_pred/current_mu)
        factors = []

        # Motion factors: use covariance scaled proportional to dt (not dt^2)
        # Qxy, Qr are std-devs; variance = Qxy**2; discrete-time covariance ~ variance * dt
        # Add small regularizer before inversion
        Q_motion = np.diag([self.Qxy**2, self.Qxy**2, self.Qr**2]) * max(dt, 1e-6)
        reg = 1e-9 * np.eye(3)
        invQ = np.linalg.inv(Q_motion + reg)

        for i in range(n):
            # delta is control integrated over dt in body frame; linearization around current_mu
            delta = uNois[:, i] * dt
            # pass the predicted prior mean to the motion factor so it can linearize about it
            f = MotionFactor(f'motion{i}', self.variables[i], current_mu[i].copy(), delta, invQ)
            factors.append(f)

        # Range factors: create range factors only for valid z entries
        R_var = max(self.Rd**2, 1e-8)
        for i in range(n):
            for j in range(i + 1, n):
                z = zNois[i, j]
                if z is None:
                    continue
                # treat NaN or nonpositive as missing measurement
                if isinstance(z, float) and (np.isnan(z) or z <= 0):
                    continue
                # pass current_mu for linearization
                f = RangeFactor(f'range{i}_{j}', self.variables[i], self.variables[j], float(z), R_var, current_mu)
                factors.append(f)

        # 4) Loopy Gaussian Belief Propagation (damped)
        for it in range(self.max_iter):
            max_diff = 0.0

            # Factor -> Variable messages (compute for all factors)
            for f in factors:
                for var in f.vars:
                    eta_new, Lambda_new = f.compute_msg_to_var(var, current_mu)
                    # damping w.r.t. previous message from same factor
                    old_eta, old_Lambda = var.incoming_messages.get(f.name, (np.zeros(3), np.zeros((3,3))))
                    eta_damped = (1 - self.beta) * old_eta + self.beta * eta_new
                    Lambda_damped = (1 - self.beta) * old_Lambda + self.beta * Lambda_new
                    var.incoming_messages[f.name] = (eta_damped, Lambda_damped)

            # Variable beliefs update (and measure change)
            for i in range(n):
                var = self.variables[i]
                mu_old = current_mu[i].copy()
                mu_new, _ = var.get_belief()
                current_mu[i] = mu_new
                diff = np.max(np.abs(mu_new - mu_old))
                max_diff = max(max_diff, diff)

            if max_diff < 1e-4:
                break

        # 5) Save results into abs_poses and convert to relativeState
        # current_mu is dict indexed by integer
        for i in range(n):
            self.abs_poses[:, i] = current_mu[i]

        # produce relativeState in each robot's body frame
        for i in range(n):
            xi = self.abs_poses[:, i]
            for j in range(n):
                if i == j:
                    continue
                xj = self.abs_poses[:, j]
                dx = xj[0] - xi[0]
                dy = xj[1] - xi[1]
                c, s = np.cos(xi[2]), np.sin(xi[2])
                relativeState[0, i, j] =  c * dx + s * dy
                relativeState[1, i, j] = -s * dx + c * dy
                # normalize yaw difference into [-pi,pi]
                dYaw = xj[2] - xi[2]
                relativeState[2, i, j] = (dYaw + np.pi) % (2*np.pi) - np.pi

        return relativeState


# ------------------- Updated helper factor classes -------------------

class MotionFactor:
    def __init__(self, name, var, mu_prior, delta, invQ):
        """
        mu_prior: 3-vector, predicted prior mean for this variable (used as linearization point)
        delta: integrated control in body frame (3-vector)
        invQ: information matrix for the motion factor (3x3)
        """
        self.name = name
        self.vars = [var]
        self.mu0 = mu_prior.copy()
        self.delta = delta.copy()
        self.invQ = invQ

    def compute_msg_to_var(self, var, current_mu_dict):
        """
        Return (eta, Lambda) for this factor -> var.
        Use current linearization point self.mu0, but optionally use current_mu_dict[var.id]
        for small correction — self.mu0 is already derived from predicted state.
        """
        # linearize around mu0
        yaw = self.mu0[2]
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy, dyaw = self.delta
        mu_prior = self.mu0 + np.array([c*dx - s*dy, s*dx + c*dy, dyaw])
        # information form: Lambda = invQ, eta = invQ @ mu_prior
        eta = self.invQ @ mu_prior
        Lambda = self.invQ.copy()
        return eta, Lambda


class RangeFactor:
    def __init__(self, name, var1, var2, z_meas, R_var, current_mu):
        """
        z_meas: scalar range measurement
        R_var: scalar observation variance
        current_mu: dict of current mu for linearization (index->3-vector)
        """
        self.name = name
        self.vars = [var1, var2]
        self.z_meas = float(z_meas)
        self.invR = 1.0 / max(R_var, 1e-12)
        self.current_mu = current_mu  # reference for linearization

    def compute_msg_to_var(self, var, current_mu_dict):
        """
        Compute message from this pairwise range factor to var (Gaussian info vector & matrix).
        Uses Schur complement / elimination to produce marginal message for var given current estimate of the other.
        """
        # identify the other variable
        v1, v2 = self.vars
        is_var1 = (var == v1)
        other_var = v2 if is_var1 else v1
        id_a = var.id
        id_b = other_var.id

        # linearize about current estimates (use provided current_mu for consistency)
        mu_a = current_mu_dict[id_a]
        mu_b = current_mu_dict[id_b]

        dx = mu_a[0] - mu_b[0]
        dy = mu_a[1] - mu_b[1]
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            dist = 1e-6

        # Jacobians of h(x_a,x_b)=||p_a-p_b|| wrt x_a and x_b
        Ja = np.array([dx / dist, dy / dist, 0.0])
        Jb = -Ja

        # build joint information (approx) on [xa; xb]: Lambda_joint = (1/R) * [Ja; Jb][Ja; Jb]^T
        J_joint = np.concatenate((Ja, Jb))  # length 6
        Lambda_joint = self.invR * np.outer(J_joint, J_joint)  # 6x6

        # partition into blocks for a and b
        Lambda_aa = Lambda_joint[0:3, 0:3]
        Lambda_ab = Lambda_joint[0:3, 3:6]
        Lambda_ba = Lambda_joint[3:6, 0:3]
        Lambda_bb = Lambda_joint[3:6, 3:6]

        # compute "eta" (information vector) for joint from linearized residual
        # Correct formula: eta = Lambda * mu_linearization + H^T * invR * (z - h(mu))
        residual = self.z_meas - dist
        mu_joint = np.concatenate((mu_a, mu_b))  # 6-vector
        eta_joint = Lambda_joint @ mu_joint + self.invR * residual * J_joint  # 6-vector
        eta_a = eta_joint[0:3]
        eta_b = eta_joint[3:6]

        # incorporate other_var's current incoming messages (information form) as prior for elimination
        eta_other, Lambda_other = other_var.get_msg_to_factor(self.name)
        # If other has no info, Lambda_other may be zero -> add tiny reg to avoid singular
        Lambda_bb_total = Lambda_bb + Lambda_other + 1e-9 * np.eye(3)

        # Schur complement: eliminate xb to get marginal for xa
        try:
            inv_bb = np.linalg.inv(Lambda_bb_total)
        except np.linalg.LinAlgError:
            inv_bb = np.linalg.pinv(Lambda_bb_total)

        Lambda_m = Lambda_aa - Lambda_ab @ inv_bb @ Lambda_ba
        eta_m = eta_a - Lambda_ab @ inv_bb @ (eta_b + eta_other)

        # ensure symmetry and small regularization
        Lambda_m = 0.5 * (Lambda_m + Lambda_m.T) + 1e-9 * np.eye(3)

        return eta_m, Lambda_m
