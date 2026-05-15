#!/usr/bin/env python3
import rospy
from rbkairos_etf_services.srv import ActionServer, ActionServerRequest

def run_my_action(coordinates):
    
    rospy.wait_for_service('/action_server')
    try:
        service_call = rospy.ServiceProxy('/action_server', ActionServer)
        
        # Format: [x, y, z, R, P, Y]

        # 10.3cm to z axis to accommodate for EE->link8:
        my_coords = [coordinates[0]-0.103,coordinates[1],coordinates[2], 2.104, 0.850, 1.867] 
        rospy.loginfo("starting grasp action")
        req = ActionServerRequest()
        req.action_id = "grasp_fruit" 
        req.input = my_coords
        req.timeout = 10.0
        
        rospy.loginfo(f"{req.action_id} : {my_coords}")
        resp = service_call(req)        
        rospy.loginfo("Grasp successful, loading to bin...")
        req_bin = ActionServerRequest()
        req_bin.action_id = "load_to_bin"
        req_bin.input = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        req_bin.timeout = 20.0
        
        resp2 = service_call(req_bin)
        rospy.loginfo("Loading to bin successful")
        return resp2

    except rospy.ServiceException as e:
        rospy.logerr(f"action call failed: {e}")

if __name__ == "__main__":
    rospy.init_node("run_grasp_action_test")
    test_coords = [0.5, 0.00, 0.5]
    run_my_action(test_coords)
    rospy.loginfo("Import this file from detect_and_grasp.py")