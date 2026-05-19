#!/usr/bin/env python3
import rospy
import sys
import tty
import termios
import threading
from std_msgs.msg import Bool

state = False

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def key_listener():
    global state
    rospy.loginfo("Ready. Press 's' = SUCCESS, 'f' = FAILURE, 'q' = quit.")
    while not rospy.is_shutdown():
        key = get_key()
        if key == 's':
            state = True
            rospy.loginfo("State: SUCCESS (True)")
        elif key == 'f':
            state = False
            rospy.loginfo("State: FAILURE (False)")
        elif key == 'q':
            rospy.signal_shutdown("User quit")
            break

def main():
    rospy.init_node("action_feedback_publisher", anonymous=False)
    pub = rospy.Publisher("/action_feedback", Bool, queue_size=1)

    t = threading.Thread(target=key_listener)
    t.daemon = True
    t.start()

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        pub.publish(Bool(data=state))
        rate.sleep()

if __name__ == "__main__":
    main()
