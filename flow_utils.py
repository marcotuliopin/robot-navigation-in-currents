import numpy as np
from queue import PriorityQueue
from scipy import interpolate


def determine_current_field(point, vortices):
    """
    Calculate the flow field at a given point based on the positions of the vortices.

    Parameters:
    point (numpy.ndarray): The point at which to calculate the flow field.
    vortices (list): A list of vortices, where each vortex is represented by its center, decay, and magnitude.

    Returns:
    numpy.ndarray: The flow field at the given point.
    """
    init = True
    for vortex in vortices:

        vortex_center, decay, magnitude = vortex
        distance = np.linalg.norm(point - vortex_center, axis=1)

        velocity_x = (
            -magnitude
            * ((point[:, 1] - vortex_center[1]) / (2 * np.pi * distance**2))
            * (1 - np.exp(-(distance**2 / decay**2)))
        )
        velocity_y = (
            magnitude
            * ((point[:, 0] - vortex_center[0]) / (2 * np.pi * distance**2))
            * (1 - np.exp(-(distance**2 / decay**2)))
        )
        Vaux = np.column_stack((velocity_x, velocity_y))

        if init:
            V = Vaux
            init = False
        else:
            V += Vaux
    return V


def calculate_weights(grid_shape, vortices, goal):
    """
    Calculate the weights for each cell in the grid based on the environmental flows and the goal.

    Parameters:
    grid_shape (tuple): The shape of the grid.
    vortices (list): A list of vortices, where each vortex is represented by its center, decay, and magnitude.
    goal (numpy.ndarray): The coordinates of the goal.

    Returns:
    tuple: A tuple containing the weights array and the forces array.
    """
    weights = np.zeros(grid_shape)
    forces = np.zeros((grid_shape[0], grid_shape[1], 2))

    for iy, ix in np.ndindex(weights.shape):
        point = np.array([ix, iy])

        # Calculate the local flow force
        flow = determine_current_field(np.array([point]), vortices)
        flow = np.squeeze(flow)
        forces[ix, iy] = flow

        # Get the direction of the goal and of the local flow force
        goal_vector = (goal - point) / np.linalg.norm(goal - point)
        flow_normalized = flow / np.linalg.norm(flow)

        # Angle between the goal vector and the local flow force
        cossine = (
            np.dot(goal_vector, flow_normalized)
            / np.linalg.norm(goal_vector)
            * np.linalg.norm(flow_normalized)
        )
        angle = np.arccos(cossine)

        # Get the projection of the flow force in the goal vector
        flow_projection = project_vector(flow, goal)

        # Add a penalty for being close to a vortex center
        dist_to_vortex = calculate_distance_to_nearest_vortex(point, vortices)
        sigma = 10
        vortex_penalty = np.exp(-(dist_to_vortex**2) / (2 * sigma**2))

        # Add a penalty for movements against the flow
        against_flow_penalty = 0
        if angle > np.pi / 2:  # The move is against the flow
            against_flow_penalty = 10

        weight = (
            np.exp(1 + angle)
            * np.linalg.norm(flow)
            / (1 + np.exp(np.linalg.norm(flow_projection)))
            + 100 * vortex_penalty
            + against_flow_penalty
        )
        weights[iy, ix] = weight

    weights = np.where(np.isnan(weights), 1, weights)
    smw = weights.min()
    weights = weights + np.abs(smw)
    return weights, forces


def djikstra(grid_shape, weights, forces, katt):
    """
    Implement the Dijkstra's algorithm to find the shortest path
    from the start to each cell in the grid.

    Parameters:
    grid_shape (tuple): The shape of the grid.
    weights (numpy.ndarray): The weights for each cell in the grid.
    forces (numpy.ndarray): The forces at each cell in the grid.
    katt (float): The attraction constant.

    Returns:
    numpy.ndarray: An array that stores the previous cell for each cell in the grid.
    """
    prev = np.zeros((grid_shape[0], grid_shape[1], 2))
    d = np.zeros(grid_shape) + np.inf
    # Define possible directions of movement
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]

    # Set start position and initial distance
    start = (0, 0)
    d[start] = 0
    prev[start] = (-1, -1)

    pq = PriorityQueue()
    pq.put((0, start))

    while not pq.empty():
        du, u = pq.get()

        if du > d[u]:
            continue

        # Calculate normalized local force
        prev_force = forces[u[0], u[1]]
        normalized_prev_force = prev_force / np.linalg.norm(prev_force)

        for dx, dy in directions:
            nx, ny = u[0] + dx, u[1] + dy

            # Skip if new position is out of bounds
            if not (0 <= nx < weights.shape[0] and 0 <= ny < weights.shape[1]):
                continue

            neighbor_direction = np.array([dx, dy]) / np.sqrt(dx**2 + dy**2)

            momentum_gain_ratio = np.dot(normalized_prev_force, neighbor_direction)
            momentum_gain = momentum_gain_ratio * weights[nx, ny]

            edge_angle = np.arccos(np.dot(neighbor_direction, normalized_prev_force))
            edge_force = np.exp(1 + edge_angle) * np.linalg.norm(prev_force)

            att_magnitude = max(katt - np.linalg.norm(prev_force), 0)

            dnew = d[u] + weights[nx, ny] + edge_force - momentum_gain - att_magnitude
            dnew = max(dnew, d[u])

            if d[nx, ny] > dnew:
                d[nx, ny] = dnew
                pq.put((d[nx, ny], (nx, ny)))
                prev[nx, ny] = u
    return prev


def create_path(grid_shape, prev, vortices, XY):
    """
    Create the path from the start to the goal using the 'prev' array 
    and interpolate it for smoother transitions.

    Parameters:
    grid_shape (tuple): The shape of the grid.
    prev (numpy.ndarray): An array that stores the previous cell for each cell in the grid.
    vortices (list): A list of vortices, where each vortex is represented by its center, decay, and magnitude.
    XY (numpy.ndarray): The coordinates of all points in the grid.

    Returns:
    numpy.ndarray: The interpolated path from the start to the goal.
    """
    path = []
    v = (grid_shape[0] - 1, grid_shape[1] - 1)

    while v[0] != -1 and v[1] != -1:
        path.append(v)
        v = tuple(prev[v].astype(int))
    path = path[::-1]
    x_coords, y_coords = zip(*path)

    V = determine_current_field(XY, vortices)
    Vx = V[:, 0]
    Vy = V[:, 1]
    M = np.hypot(Vx, Vy)
    tck, u = interpolate.splprep([x_coords, y_coords], s=200)
    unew = np.linspace(0, 1.0, 1000)
    out = interpolate.splev(unew, tck)

    path = np.stack(out).T
    return path


def calculate_distance_to_nearest_vortex(point, vortices):
    min_distance = np.inf
    for vortex in vortices:
        vortex_center, _, _ = vortex
        vortex_point_distance = np.linalg.norm(vortex_center - point)
        min_distance = min(vortex_point_distance, min_distance)
    return min_distance


def project_vector(u, v):
    return np.dot(u, v) / np.linalg.norm(v) ** 2 * v
