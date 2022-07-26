#!/usr/bin/env python

import os, sys
import rospy
import rospkg
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget, QApplication
from gui.msg import AttackCmd

class AttackUi(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._rp = rospkg.RosPack()
        self._rp_package_list = self._rp.list()
        res_folder = os.path.join(self._rp.get_path('gui'), 'ui')
        ui_file = os.path.join(res_folder, 'attack.ui')
        loadUi(ui_file, self)

        self.launch_button.clicked.connect(self.launch_click)
        # attack_types = ['0', '1']
        # self.attack_type.addItems(attack_types)

        self.pub = rospy.Publisher('attack_cmd', AttackCmd, queue_size=10)
        rospy.init_node('attack_controller', anonymous=True)

        self.show()

    def launch_click(self):
        cmd = AttackCmd()
        cmd.b0e = self.b0e.isChecked()
        cmd.b1e = self.b1e.isChecked()
        cmd.b2e = self.b2e.isChecked()
        cmd.b3e = self.b3e.isChecked()
        cmd.b0 = self.b0.value() if cmd.b0e else 0
        cmd.b1 = self.b1.value() if cmd.b1e else 0 
        cmd.b2 = self.b2.value() if cmd.b2e else 0
        cmd.b3 = self.b3.value() if cmd.b3e else 0
        cmd.attack_duration = self.attack_index.value()
        cmd.detection_delay = self.recovery_index.value()
        self.pub.publish(cmd)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = AttackUi()
    sys.exit(app.exec_())


