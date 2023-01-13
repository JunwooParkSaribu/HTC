class H2B():
    def __init__(self):
        self.trajectory = []
        self.time = []
        self.max_radius = None

    def get_trajectory(self):
        return self.trajectory

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory

    def get_time(self):
        return self.time

    def set_time(self, time):
        self.time = time

    def get_len_trajectory(self):
        return len(self.trajectory)

    def get_time_duration(self):
        return self.time[-1] - self.time[0]

    def set_max_radius(self, radius):
        self.max_radius = radius

    def get_max_radius(self):
        return self.max_radius
