from math import sin, cos, pi

class GenerateTrack:
    def __init__(self, output_file):
        self.output = output_file

    def straight(self, start, end, num):
        dx = (end[0] - start[0])/num
        dy = (end[1] - start[1])/num
        x = start[0]
        y = start[1]
        with open(self.output, 'a') as o:
            for i in range(num):
                x += dx
                y += dy
                o.write(f'{x:.2f}  {y:.2f}\n')

    def circle(self, center, radius, angle_range, num):
        a = angle_range[0]
        da = (angle_range[1] - angle_range[0]) / num
        with open(self.output, 'a') as o:
            for i in range(num):
                a += da
                x = center[0] + radius * cos(a)
                y = center[1] + radius * sin(a)
                o.write(f'{x:.2f}  {y:.2f}\n')


if __name__ == '__main__':
    filename = 'cubetown_close_track.txt'
    t = GenerateTrack(filename)
    # half
    # t.straight(start=(35, 3.15), end=(-35, 3.15), num=200)
    # t.circle(center=(-35, 18.6), radius=15.45, angle_range=(3*pi/2, pi), num=50)
    # t.straight(start=(-50.45, 18.6), end=(-50.45, 35), num=100)
    # t.circle(center=(-35, 35), radius=15.45, angle_range=(pi, pi/2), num=50)
    # t.straight(start=(-35, 50.45), end=(35, 50.45), num=200)
    # t.circle(center=(35, 35), radius=15.45, angle_range=(pi/2, 0), num=50)
    # t.straight(start=(50.45, 35), end=(50.45, 18.6), num=100)
    # t.circle(center=(35, 18.6), radius=15.45, angle_range=(0, -pi/2), num=50)

    t.straight(start=(35, 3.15), end=(-35, 3.15), num=200)
    t.circle(center=(-35, 18.6), radius=15.45, angle_range=(3*pi/2, pi), num=50)
    t.straight(start=(-50.45, 18.6), end=(-50.45, 35), num=100)
    t.circle(center=(-35, 35), radius=15.45, angle_range=(pi, pi/2), num=50)
    t.straight(start=(-35, 50.45), end=(35, 50.45), num=200)
    t.circle(center=(35, 35), radius=15.45, angle_range=(pi/2, 0), num=50)
    t.straight(start=(50.45, 35), end=(50.45, -35), num=100)
    t.circle(center=(35, -35), radius=15.45, angle_range=(0, -pi/2), num=50)
    t.straight(start=(35, -50.45), end=(-35, -50.45), num=100)
    t.circle(center=(-35, -35), radius=15.45, angle_range=(3*pi/2, pi), num=50)
    t.straight(start=(-50.45, -35), end=(-50.45, 18.6), num=110)
