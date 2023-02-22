import numpy as np
from H2B import H2B


def make_immobile(histones, histones_label, nb=5, radius=0.45, max_distance=0.05, cond=(100, 150)):
    for i in range(nb):
        h2b = H2B()
        n_trajectory = int(np.random.uniform(cond[0], cond[1]))
        trajectory = [[10, 10]]
        prev_xy = trajectory[0]

        while len(trajectory) < n_trajectory:
            x = np.random.uniform(prev_xy[0]-max_distance, prev_xy[0]+max_distance)
            y = np.random.uniform(prev_xy[1]-max_distance, prev_xy[1]+max_distance)
            xy = np.array([x, y])
            if np.sqrt((xy[0] - trajectory[0][0])**2 + (xy[1] - trajectory[0][1])**2) < radius:
                if len(trajectory) > 1:
                    direction = xy - prev_xy
                    if np.dot(direction, prev_direction) < 0:
                        trajectory.append(xy)
                        prev_direction = direction
                        prev_xy = xy
                else:
                    trajectory.append(xy)
                    prev_direction = xy - prev_xy
                    prev_xy = xy

        trajectory = np.array(trajectory)
        times = (np.arange(n_trajectory)+1)*0.01
        h2b.set_trajectory(trajectory)
        h2b.set_time(times)
        h2b.set_file_name('simulated_data')
        h2b.set_manuel_label(0)

        if len(histones) != 0:
            id = str(np.sort(np.array(list(histones.keys())).astype(int))[-1] + 1)
            h2b.set_id(id)
            histones[id] = h2b
            histones_label[id] = 0
        else:
            h2b.set_id(str(0))
            histones[str(0)] = h2b
            histones_label[str(0)] = 0


def make_mobile(histones, histones_label, nb=5, max_distance=0.5, cond=(30, 50)):
    for i in range(nb):
        h2b = H2B()
        n_trajectory = int(np.random.uniform(cond[0], cond[1]))
        trajectory = [[10, 10]]
        prev_xy = trajectory[0]

        while len(trajectory) < n_trajectory:
            x = np.random.uniform(prev_xy[0]-max_distance, prev_xy[0]+max_distance)
            y = np.random.uniform(prev_xy[1]-max_distance, prev_xy[1]+max_distance)
            xy = np.array([x, y])
            trajectory.append(xy)
            prev_xy = xy

        trajectory = np.array(trajectory)
        times = (np.arange(n_trajectory)+1)*0.01
        h2b.set_trajectory(trajectory)
        h2b.set_time(times)
        h2b.set_file_name('simulated_data')
        h2b.set_manuel_label(2)

        if len(histones) != 0:
            id = str(np.sort(np.array(list(histones.keys())).astype(int))[-1] + 1)
            h2b.set_id(id)
            histones[id] = h2b
            histones_label[id] = 2
        else:
            h2b.set_id(str(0))
            histones[str(0)] = h2b
            histones_label[str(0)] = 2
    return histones


def make_hybrid(histones, histones_label, nb=5, radius=0.45, max_dist_immobile=0.01, max_dist_mobile=0.3, type=0):
    for i in range(nb):
        h2b = H2B()
        n_trajectory = int(np.random.uniform(100, 150))
        intermediate_trajectory = int(np.random.randint(5, 20))
        trajectory = [[10, 10]]
        prev_xy = trajectory[0]

        while len(trajectory) < n_trajectory:
            if type == 0:
                if len(trajectory) < n_trajectory/3:
                    x = np.random.uniform(prev_xy[0]-max_dist_immobile, prev_xy[0]+max_dist_immobile)
                    y = np.random.uniform(prev_xy[1]-max_dist_immobile, prev_xy[1]+max_dist_immobile)
                    xy = np.array([x, y])
                    if np.sqrt((xy[0] - trajectory[0][0])**2 + (xy[1] - trajectory[0][1])**2) < radius:
                        if len(trajectory) > 1:
                            direction = xy - prev_xy
                            if np.dot(direction, prev_direction) < 0:
                                trajectory.append(xy)
                                prev_direction = direction
                                prev_xy = xy
                        else:
                            trajectory.append(xy)
                            prev_direction = xy - prev_xy
                            prev_xy = xy
                elif len(trajectory) < (n_trajectory/3 + intermediate_trajectory):
                    x = np.random.uniform(prev_xy[0] - max_dist_mobile, prev_xy[0] + max_dist_mobile)
                    y = np.random.uniform(prev_xy[1] - max_dist_mobile, prev_xy[1] + max_dist_mobile)
                    xy = np.array([x, y])
                    trajectory.append(xy)
                    prev_direction = xy - prev_xy
                    prev_xy = xy
                else:
                    new_center = prev_xy
                    x = np.random.uniform(prev_xy[0]-max_dist_immobile, prev_xy[0]+max_dist_immobile)
                    y = np.random.uniform(prev_xy[1]-max_dist_immobile, prev_xy[1]+max_dist_immobile)
                    xy = np.array([x, y])
                    if np.sqrt((xy[0] - new_center[0])**2 + (xy[1] - new_center[1])**2) < radius:
                        if len(trajectory) > 1:
                            direction = xy - prev_xy
                            if np.dot(direction, prev_direction) < 0:
                                trajectory.append(xy)
                                prev_direction = direction
                                prev_xy = xy

            elif type == 1:
                if len(trajectory) < 9*n_trajectory/10:
                    x = np.random.uniform(prev_xy[0]-max_dist_immobile, prev_xy[0]+max_dist_immobile)
                    y = np.random.uniform(prev_xy[1]-max_dist_immobile, prev_xy[1]+max_dist_immobile)
                    xy = np.array([x, y])
                    if np.sqrt((xy[0] - trajectory[0][0])**2 + (xy[1] - trajectory[0][1])**2) < radius:
                        if len(trajectory) > 1:
                            direction = xy - prev_xy
                            if np.dot(direction, prev_direction) < 0:
                                trajectory.append(xy)
                                prev_direction = direction
                                prev_xy = xy
                        else:
                            trajectory.append(xy)
                            prev_direction = xy - prev_xy
                            prev_xy = xy
                else:
                    x = np.random.uniform(prev_xy[0] - max_dist_mobile, prev_xy[0] + max_dist_mobile)
                    y = np.random.uniform(prev_xy[1] - max_dist_mobile, prev_xy[1] + max_dist_mobile)
                    xy = np.array([x, y])
                    trajectory.append(xy)
                    prev_direction = xy - prev_xy
                    prev_xy = xy

            else:
                if len(trajectory) < n_trajectory/10:
                    x = np.random.uniform(prev_xy[0] - max_dist_mobile, prev_xy[0] + max_dist_mobile)
                    y = np.random.uniform(prev_xy[1] - max_dist_mobile, prev_xy[1] + max_dist_mobile)
                    xy = np.array([x, y])
                    trajectory.append(xy)
                    prev_direction = xy - prev_xy
                    prev_xy = xy
                else:
                    new_center = prev_xy
                    x = np.random.uniform(prev_xy[0]-max_dist_immobile, prev_xy[0]+max_dist_immobile)
                    y = np.random.uniform(prev_xy[1]-max_dist_immobile, prev_xy[1]+max_dist_immobile)
                    xy = np.array([x, y])
                    if np.sqrt((xy[0] - new_center[0])**2 + (xy[1] - new_center[1])**2) < radius:
                        if len(trajectory) > 1:
                            direction = xy - prev_xy
                            if np.dot(direction, prev_direction) < 0:
                                trajectory.append(xy)
                                prev_direction = direction
                                prev_xy = xy

        trajectory = np.array(trajectory)
        times = (np.arange(n_trajectory)+1)*0.01
        h2b.set_trajectory(trajectory)
        h2b.set_time(times)
        h2b.set_file_name('simulated_data')
        h2b.set_manuel_label(1)

        if len(histones) != 0:
            id = str(np.sort(np.array(list(histones.keys())).astype(int))[-1] + 1)
            h2b.set_id(id)
            histones[id] = h2b
            histones_label[id] = 1
        else:
            h2b.set_id(str(0))
            histones[str(0)] = h2b
            histones_label[str(0)] = 1


def make_simulation_data(number=3000):
    histones = {}
    histones_label = {}

    # make immobile H2Bs
    for dist_immo in range(2, 7):
        dist_immo /= 100
        make_immobile(histones, histones_label, nb=int(number/5), radius=0.45, max_distance=dist_immo)

    # make hybrid H2Bs
    for dist_immo in range(3, 8):
        dist_immo /= 100
        for dist_mob in range(2, 4):
            dist_mob /= 10
            make_hybrid(histones, histones_label, nb=int(number/30), radius=0.45, max_dist_immobile=dist_immo,
                        max_dist_mobile=dist_mob, type=0)
            make_hybrid(histones, histones_label, nb=int(number/30), radius=0.45, max_dist_immobile=dist_immo,
                        max_dist_mobile=dist_mob, type=1)
            make_hybrid(histones, histones_label, nb=int(number/30), radius=0.45, max_dist_immobile=dist_immo,
                        max_dist_mobile=dist_mob, type=2)

    # make mobile H2Bs
    for dist_mob in range(3, 6):
        dist_mob /= 10
        make_mobile(histones, histones_label, nb=int(number/3), max_distance=dist_mob)
    return histones, histones_label
