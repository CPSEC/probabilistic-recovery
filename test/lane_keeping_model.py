import numpy as np

cf_ = 155494.663
cr_ = 155494.663
# cf_ = -155494.663
# cr_ = -155494.663
wheelbase_ = 2.852
steer_ratio_ = 16
steer_single_direction_max_degree_ = 470.0
mass_fl = 520
mass_fr = 520
mass_rl = 520
mass_rr = 520
mass_front = mass_fl + mass_fr
mass_rear = mass_rl + mass_rr
mass_ = mass_front + mass_rear
lf_ = wheelbase_ * (1.0 - mass_front / mass_)
lr_ = wheelbase_ * (1.0 - mass_rear / mass_)
iz_ = lf_ * lf_ * mass_front + lr_ * lr_ * mass_rear


class LaneKeeping:
    def __init__(self, vx) -> None:
        self.A = np.zeros((4, 4), dtype=np.float32)
        self.A[0, 1] = 1.0
        self.A[1, 2] = (cf_ + cr_) / mass_
        self.A[2, 3] = 1.0
        self.A[3, 2] = (lf_ * cf_ - lr_ * cr_) / iz_

        self.B = np.zeros((4, 1), dtype=np.float32)
        self.B[1, 0] = cf_ / mass_
        self.B[3, 0] = lf_ * cf_ / iz_

        self.C = np.eye(4)
        self.D = np.zeros((4, 1))

        self.update(vx)

    def update(self, vx):
        self.A[1, 1] = -(cf_ + cr_) / mass_ / vx
        self.A[1, 3] = (lr_ * cr_ - lf_ * cf_) / mass_ / vx
        self.A[3, 1] = (lr_ * cr_ - lf_ * cf_) / iz_ / vx
        self.A[3, 3] = -1.0 * (lf_ * lf_ * cf_ + lr_ * lr_ * cr_) / iz_ / vx

# class LaneKeeping:
#     def __init__(self, vx) -> None:
#         self.A = np.zeros((4, 4), dtype=np.float32)
#         self.A[0, 1] = 1.0
#         self.A[1, 2] = -(cf_ + cr_) / mass_
#         self.A[2, 3] = 1.0
#         self.A[3, 2] = -(lf_ * cf_ - lr_ * cr_) / iz_

#         self.B = np.zeros((4, 1), dtype=np.float32)
#         self.B[1, 0] = -cf_ / mass_
#         self.B[3, 0] = -lf_ * cf_ / iz_

#         self.C = np.eye(4)
#         self.D = np.zeros((4, 1))

#         self.update(vx)

#     def update(self, vx):
#         self.A[1, 1] = (cf_ + cr_) / mass_ / vx
#         self.A[1, 3] = (lf_ * cf_ - lr_ * cr_ ) / mass_ / vx
#         self.A[3, 1] = (lf_ * cf_ - lr_ * cr_ ) / iz_ / vx
#         self.A[3, 3] = (lf_ * lf_ * cf_ + lr_ * lr_ * cr_) / iz_ / vx