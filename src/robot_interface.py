import serial
import time
from config import SERIAL_PORT, BAUDRATE

class RobotInterface:
    def __init__(self, port=SERIAL_PORT, baud=BAUDRATE):
        self.ser = None
        if port:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)

    def send(self, linear_vel, angular_vel):
        """
        Gửi lệnh đến robot: - chưa dùng :V
        Format: "L<linear>,A<angular>\n"
        Nếu không có serial, chỉ in ra console.
        """
        if self.ser:
            cmd = f"L{linear_vel:.2f},A{angular_vel:.2f}\n"
            self.ser.write(cmd.encode('utf-8'))
        else:
            print(f"[ROBOT CMD] linear={linear_vel:.2f}, angular={angular_vel:.2f}")

    def close(self):
        if self.ser:
            self.ser.close()
