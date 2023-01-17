from time import perf_counter

class Timer:
    def __init__(self):
        self.total_time = 0
        self.tic()

    def tic(self):
        self.start = perf_counter()

    def toc(self):
        self.total_time += perf_counter() - self.start

    def reset(self):
        self.total_time = 0
        self.tic()

    def total(self):
        return self.total_time
