import numpy as np
import matplotlib.pyplot as plt
from flow_utils import *
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Link to simulator API
client = RemoteAPIClient()
sim = client.require("sim")
sim.setStepping(False)

# Constants
WORLDX, WORLDY = 100, 100
step = 5
katt = 0.75
R = 10
robot_path = "/PioneerP3DX"
robot_handle = sim.getObject("/PioneerP3DX")

rmotor = sim.getObject(robot_path + "/rightMotor")
lmotor = sim.getObject(robot_path + "/leftMotor")

kr = 1
kt = 0.7
L = 0.381
r = 0.0975
maxv = 1.0
maxw = np.deg2rad(45)

# Define vortices
vortices = [
    [np.array([20, 20]), 10, 100],
    [np.array([20, 80]), 10, 100],
    [np.array([80, 20]), 10, -100],
    [np.array([50, 40]), 10, 200],
]

# Create grid
XX, YY = np.meshgrid(
    np.arange(0, WORLDX + step, step), np.arange(0, WORLDY + step, step)
)
XY = np.dstack([XX, YY]).reshape(-1, 2)

# Define start and goal positions
start = np.array([0, 0])
goal = np.array([WORLDX, WORLDY])
grid_shape = (100, 100)

weights, forces = calculate_weights(grid_shape, vortices, goal)
prev = djikstra(grid_shape, weights, forces, katt)
path = create_path(grid_shape, prev, vortices, XY)

# Robot start position
q = np.array([0, 0])

target_index = 100
target = path[target_index]

# History list
hist = []

# Time and distance to goal
t = 0
rho = np.inf

fig = plt.figure(figsize=(8, 5), dpi=100)
ax = fig.add_subplot(111, aspect="equal")

sim.startSimulation()
sim.setJointTargetVelocity(rmotor, 0)
sim.setJointTargetVelocity(lmotor, 0)

while rho > R:
    *pcurr, _ = sim.getObjectPosition(robot_handle)
    _, _, ocurr = sim.getObjectOrientation(robot_handle)

    q = np.array(pcurr[:2]) * 4

    ax.cla()

    # Check if the robot has reached the current target
    if np.linalg.norm(target - q) < R:
        # Move to the next target on the path
        target_index += 50
        if target_index >= len(path):
            target = goal
        else:
            target = path[target_index]

    # Flow
    fl = determine_current_field(np.array([q]), vortices)

    # Calculate necessary attraction force to make the sum of the attraction
    # force and the flow point towards the target
    desired_direction = (target - q) / np.linalg.norm(target - q)
    at = desired_direction - fl[0]

    # Normalize attraction force to have constant magnitude
    at = (at / np.linalg.norm(at)) * katt

    # Calculate total force
    u = at + fl[0]

    fx = u[0]
    fy = u[1]

    # Control [De Luca e Oriolo, 1994]
    v = kr * (fx * np.cos(ocurr) + fy * np.sin(ocurr))
    w = kt * (np.arctan2(fy, fx) - ocurr)

    v = max(min(v, maxv), -maxv)
    w = max(min(w, maxw), -maxw)

    vr = ((2.0 * v) + (w * L)) / (2.0 * r)
    vl = ((2.0 * v) - (w * L)) / (2.0 * r)
    sim.setJointTargetVelocity(rmotor, vr)
    sim.setJointTargetVelocity(lmotor, vl)

    hist.append(q)

    # PLOTS
    ax.scatter(*goal, s=50, c="g", marker="X", zorder=5)

    V = determine_current_field(XY, vortices)

    Vx = V[:, 0]
    Vy = V[:, 1]
    M = np.hypot(Vx, Vy)

    ax.quiver(XX, YY, Vx, Vy, M, cmap="cool", scale=20)
    ax.plot(*q, "o", color="r")
    ax.plot(*target, "o", color="b")

    ax.quiver(*q, *at, color="g", scale=20)
    ax.quiver(*q, *u, color="b", scale=20)
    ax.quiver(*q, *fl[0], color="m", scale=20)

    h = np.array(hist)
    ax.plot(h[:, 0], h[:, 1], "k--")

    ax.set_title(f"t = {t:.1f}")

    plt.pause(0.01)

    rho = np.linalg.norm(goal - q)
    t = t + 0.1
plt.show()
