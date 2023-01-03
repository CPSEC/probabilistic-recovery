from math import atan2

class Path:
    def __init__(self, filename) -> None:
        self.x = []
        self.y = []
        self.heading = []
        self.accumulated_s = []
        self.kappa = []
        self.dkappa = []
        self.load_from_file(filename)
        self.compute_profile()

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                x, y = map(float, line.split())
                self.x.append(x)
                self.y.append(y)

    def compute_profile(self):
        k = len(self)
        # compute dx, dy, heading
        dx = []
        dy = []
        for i in range(k):
            if i==0:
                x_delta = self.x[i+1] - self.x[i]
                y_delta = self.y[i+1] - self.y[i]
            elif i==k-1:
                x_delta = self.x[i] - self.x[i-1]
                y_delta = self.y[i] - self.y[i-1]
            else:
                x_delta = (self.x[i+1] - self.x[i-1])*0.5
                y_delta = (self.y[i+1] - self.y[i-1])*0.5
            dx.append(x_delta)
            dy.append(y_delta)
            self.heading.append(atan2(y_delta, x_delta))

        # Get linear interpolated s for dkappa calculation
        distance = 0.0
        self.accumulated_s = [distance]
        fx, fy = self.x[0], self.y[0]
        for i in range(1, k):
            nx, ny = self.x[i], self.y[i]
            end_segment_s = ((fx-nx)**2 + (fy-ny)**2)**0.5
            distance += end_segment_s
            self.accumulated_s.append(distance)
            fx, fy = nx, ny
        
        # Get finite difference approximated first derivative of y and x respective
        # to s for kappa calculation
        dxds, dyds = self.derivative_pairs_over_c(self.x, self.y, self.accumulated_s)

        # Get finite difference approximated second derivative of y and x
        # respective to s for kappa calculation
        ddxds, ddyds = self.derivative_pairs_over_c(dxds, dyds, self.accumulated_s)

        # kappa
        for i in range(k):
            kappa = (dxds[i]*ddyds[i] - dyds[i]*ddxds[i]) / ( (dxds[i]**2 + dyds[i]**2)**1.5 + 1e-6)
            self.kappa.append(kappa)

        # dkappa
        dkappa, _ = self.derivative_pairs_over_c(self.kappa, self.kappa, self.accumulated_s)
        self.dkappa = dkappa


    @staticmethod
    def derivative_pairs_over_c(a, b, c):
        dadc = []
        dbdc = []
        k=len(a)
        for i in range(k):
            if i==0:
                adc = (a[i+1] - a[i]) / (c[i+1] - c[i])
                bdc = (b[i+1] - b[i]) / (c[i+1] - c[i])
            elif i==k-1:
                adc = (a[i] - a[i-1]) / (c[i] - c[i-1])
                bdc = (b[i] - b[i-1]) / (c[i] - c[i-1])
            else:
                adc = (a[i+1] - a[i-1]) / (c[i+1] - c[i-1])
                bdc = (b[i+1] - b[i-1]) / (c[i+1] - c[i-1])
            dadc.append(adc)
            dbdc.append(bdc)
        return dadc, dbdc


    def __len__(self):
        return len(self.x)

    def get_nearest_index(self, x, y):
        k = len(self)
        dmin = float('inf')
        dmin_index = 0
        for i in range(k):
            dis = (x-self.x[i])**2 + (y-self.y[i])**2
            if dis < dmin:
                dmin = dis
                dmin_index = i
        return dmin_index


    
