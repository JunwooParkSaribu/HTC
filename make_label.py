import trajectory_phy


def make_label(histones, radius=0.2, density=0.5) -> []:
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


"""
def make_label(histones, immobile_cutoff) -> []:
    distances = trajectory_phy.distance(histones)
    displacements = trajectory_phy.displacement(histones)
    histone_label = {}
    for histone in histones:
        ratio = distances[histone][0] / displacements[histone][0]
        radius, t = displacements[histone]
        if radius < immobile_cutoff and ratio < 3:
            histone_label[histone] = 0 # immobile
        elif ratio > 3 and t < 0.1:
            histone_label[histone] = 2 # mobile
        else:
            histone_label[histone] = 1 # hybrid
    return histone_label
"""
