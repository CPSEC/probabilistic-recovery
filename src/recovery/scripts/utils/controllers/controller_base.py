from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def update(self, feedback_value, current_time=None):
        pass

    @abstractmethod
    def set_control_limit(self, control_lo, control_up):
        pass

    @abstractmethod
    def set_reference(self, ref):
        pass

    def clear(self):
        pass
