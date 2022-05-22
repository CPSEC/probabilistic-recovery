class HalfSpace:
    """
    normal vector - l
    The half space is defined as l^T x > b
    """
    def __init__(self, l, b):
        self.l = l
        self.b = b
