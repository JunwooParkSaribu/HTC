class H2B():
    def __init__(self):
        self.trajectory = []
        self.time = []
        self.max_radius = None
        self.channel = []
        self.channel_size = 0

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

    def set_channel(self, channel):
        self.channel = channel

    def get_channel(self):
        return self.channel

    def set_channel_size(self, n):
        self.channel_size = n

    def get_channel_size(self):
        return self.channel_size