"""
Project: Simulation of a swarm of robots with relative localization
This version will use GBP if available, otherwise falls back to the EKF.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dataCreate import dataCreate
# try to import GBP implementation first; fallback to EKF
try:
    from relativeGBP import GBPonSimData as RelEstimatorClass
    use_gbp = True
except Exception:
    # fallback: original EKF implementation
    from relativeEKF import EKFonSimData as RelEstimatorClass
    use_gbp = False

import transform

print("Using estimator:", "GBP" if use_gbp else "EKF")

# Simulation settings
show_animation = True  # True: animation; False: figure
np.random.seed(19910620)  # reproducible
border = {"xmin": -10, "xmax": 6, "ymin": -6, "ymax": 6, "zmin": 0, "zmax": 6}
numRob = 4  # number of robots
dt = 0.01  # time interval [s]
simTime = 70.0  # simulation time [s]
maxVel = 1  # maximum velocity [m/s]
devInput = np.array([[0.25, 0.25, 0.01]]).T  # input deviation (Vx, Vy, yawRate)
devObser = 0.1  # observation deviation of distance [m]
ekfStride = 1  # update interval multiplier (ekfStride*0.01 s)

# Variables being updated in simulation
xTrue = np.random.uniform(-3, 3, (3, numRob))  # true states (x,y,yaw)
relativeState = np.zeros((3, numRob, numRob))  # x_ij, y_ij, yaw_ij in i's frame

data = dataCreate(numRob, border, maxVel, dt, devInput, devObser)

# instantiate estimator (GBP or EKF); use named args to match both constructors
estimator = RelEstimatorClass(Pxy=10.0, Pr=0.1, Qxy=0.25, Qr=0.4, Rd=0.1, numRob=numRob)

# convenience label for plotting
est_label = "Relative GBP" if use_gbp else "Relative EKF"


def animate(step):
    global xTrue, relativeState
    u = data.calcInput_FlyIn1m(step)
    #u = data.calcInput_PotentialField(step, xTrue)
    #u = data.calcInput_Formation01(step, relativeState)
    xTrue, zNois, uNois = data.update(xTrue, u)

    if step % ekfStride == 0:
        # call the right method depending on estimator type
        if use_gbp:
            # GBP signature: GBP(self, uNois, zNois, relativeState, ekfStride)
            relativeState = estimator.GBP(uNois, zNois, relativeState, ekfStride)
        else:
            # EKF signature: EKF(self, uNois, zNois, relativeState, ekfStride)
            relativeState = estimator.EKF(uNois, zNois, relativeState, ekfStride)

    # convert estimated relative states into absolute positions (w.r.t. robot 0)
    # transform.calcAbsPosUseRelaPosWRTRob0 expects (xTrue[:,0], relativeState, xTrue, numRob)
    xEsti = transform.calcAbsPosUseRelaPosWRTRob0(xTrue[:, 0], relativeState, xTrue, numRob)

    pointsTrue.set_data(xTrue[0, :], xTrue[1, :])  # ground truth
    pointsEsti.set_data(xEsti[0, :], xEsti[1, :])  # estimates
    pointsTrueHead.set_data(
        xTrue[0, :] + 0.07 * np.cos(xTrue[2, :]), xTrue[1, :] + 0.07 * np.sin(xTrue[2, :])
    )
    pointsEstiHead.set_data(
        xEsti[0, :] + 0.07 * np.cos(xEsti[2, :]), xEsti[1, :] + 0.07 * np.sin(xEsti[2, :])
    )
    circle.center = (xTrue[0, 0], xTrue[1, 0])
    # guard when zNois might be nan or missing
    try:
        circle.radius = float(zNois[0, 1])
    except Exception:
        circle.radius = 0.0
    time_text.set_text("t={:.2f}s".format(step * dt))
    return pointsTrue, pointsEsti, circle, pointsTrueHead, pointsEstiHead, time_text

fps=30
duration_seconds = 60
num_frames = fps * duration_seconds
if show_animation:
    # Set up an animation
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    ax.set(xlim=(border["xmin"], border["xmax"]), ylim=(border["ymin"], border["ymax"]))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    title = ax.set_title("Simulated swarm — estimator: {}".format(est_label))
    pointsTrue, = ax.plot([], [], linestyle="", marker="o", color="b", label="GroundTruth")
    pointsEsti, = ax.plot([], [], linestyle="", marker="o", color="r", label=est_label)
    pointsTrueHead, = ax.plot([], [], linestyle="", marker=".", color="g")
    pointsEstiHead, = ax.plot([], [], linestyle="", marker=".", color="g")
    ax.legend()
    circle = plt.Circle((0, 0), 0.2, color="black", fill=False)
    ax.add_patch(circle)
    time_text = ax.text(0.01, 0.97, "", transform=ax.transAxes)
    time_text.set_text("")
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1000/fps, blit=True)
    ani.save('particle_box.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.show()
else:
    # Batch simulation / plotting of errors
    xEsti = relativeState[:, 0, :]
    xTrueRL = transform.calcRelaState(xTrue, numRob)
    dataForPlot = np.array(
        [xEsti[0, 1], xTrueRL[0, 1], xEsti[1, 1], xTrueRL[1, 1], xEsti[2, 1], xTrueRL[2, 1]]
    )
    step = 0
    while simTime >= dt * step:
        step += 1
        u = data.calcInput_FlyIn1m(step)
        xTrue, zNois, uNois = data.update(xTrue, u)
        if step % ekfStride == 0:
            if use_gbp:
                relativeState = estimator.GBP(uNois, zNois, relativeState, ekfStride)
            else:
                relativeState = estimator.EKF(uNois, zNois, relativeState, ekfStride)

        xEsti = relativeState[:, 0, :]
        xTrueRL = transform.calcRelaState(xTrue, numRob)
        dataForPlot = np.vstack(
            [dataForPlot, np.array([xEsti[0, 1], xTrueRL[0, 1], xEsti[1, 1], xTrueRL[1, 1], xEsti[2, 1], xTrueRL[2, 1]])]
        )

    dataForPlotArray = dataForPlot.T
    timePlot = np.arange(0, len(dataForPlotArray[0])) / 100
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    plt.margins(x=0)
    ax1.plot(timePlot, dataForPlotArray[0, :])
    ax1.plot(timePlot, dataForPlotArray[1, :])
    ax1.set_ylabel(r"$x_{ij}$ (m)", fontsize=12)
    ax1.grid(True)
    ax2.plot(timePlot, dataForPlotArray[2, :])
    ax2.plot(timePlot, dataForPlotArray[3, :])
    ax2.set_ylabel(r"$y_{ij}$ (m)", fontsize=12)
    ax2.grid(True)
    ax3.plot(timePlot, dataForPlotArray[4, :], label=est_label)
    ax3.plot(timePlot, dataForPlotArray[5, :], label="Ground-truth")
    ax3.set_ylabel(r"$\mathrm{\psi_{ij}}$ (rad)", fontsize=12)
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.grid(True)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.8, 0.6), shadow=True, ncol=1, fontsize=12)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()
