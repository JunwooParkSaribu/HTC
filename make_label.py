import trajectory_phy


def make_label(histones, radius=0.35, density=0.5) -> []:
    histones_balls = trajectory_phy.check_balls(histones, radius, density)
    histone_label = {}
    for histone in histones:
        if histones_balls[histone][0] == 0:
            histone_label[histone] = 2  # mobile
        elif histones_balls[histone][1] == 1:
            histone_label[histone] = 1  # hybrid
        else:
            histone_label[histone] = 0  # immobile
    del histones_balls
    return histone_label

