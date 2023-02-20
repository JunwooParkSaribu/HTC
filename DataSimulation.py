import numpy as np
from H2B import H2B
import ImagePreprocessor

def make_immobile(histones, nb=5, radius=0.45, max_distance=0.05):
    for i in range(nb):
        h2b = H2B()
        n_trajectory = int(np.random.uniform(100, 150))
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

        if len(histones) != 0:
            id = str(np.sort(np.array(list(histones.keys())).astype(int))[-1] + 1)
            h2b.set_id(id)
            histones[id] = h2b
        else:
            h2b.set_id(str(0))
            histones[str(0)] = h2b

    return histones


histones = {}
for max_distance in range(1, 11):
    max_distance /= 100
    print(max_distance)
    histones = make_immobile(histones, nb=5, radius=0.45, max_distance=max_distance)



ImagePreprocessor.make_channel(histones, immobile_cutoff=0.3, hybrid_cutoff=10, nChannel=3)
histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(histones, img_scale=10, amp=2)
zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
for histone in histones:
    trajectory = histones[histone].get_trajectory()
    histone_first_pos = [int(trajectory[0][0] * (10 ** 2)),
                         int(trajectory[0][1] * (10 ** 2))]
    ImagePreprocessor.img_save(zoomed_imgs[histone], histones[histone], scaled_size,
                               histone_first_pos=histone_first_pos, amp=2, path='./data/SimulationData/images')

