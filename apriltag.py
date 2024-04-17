from pupil_apriltags import Detector
import cv2
import numpy as np
import time
from robomaster import robot
from robomaster import camera

at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

def find_pose_from_tag(K, detection):
    m_half_size = tag_size / 2

    marker_center = np.array((0, 0, 0))
    marker_points = []
    marker_points.append(marker_center + (-m_half_size, m_half_size, 0))
    marker_points.append(marker_center + ( m_half_size, m_half_size, 0))
    marker_points.append(marker_center + ( m_half_size, -m_half_size, 0))
    marker_points.append(marker_center + (-m_half_size, -m_half_size, 0))
    _marker_points = np.array(marker_points)

    object_points = _marker_points
    image_points = detection.corners

    pnp_ret = cv2.solvePnP(object_points, image_points, K, distCoeffs=None,flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if pnp_ret[0] == False:
        raise Exception('Error solving PnP')

    r = pnp_ret[1]
    p = pnp_ret[2]

    return p.reshape((3,)), r.reshape((3,))


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    ep_chassis = ep_robot.chassis


    tag_size=0.16 # tag size in meters

    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)  
            cv2.imwrite("/home/user/Desktop/test.png", img) 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray.astype(np.uint8)

            K=np.array([[184.752, 0, 320], [0, 184.752, 180], [0, 0, 1]])

            results = at_detector.detect(gray, estimate_tag_pose=False)
            
            for res in results:
                # print(res)
                pose = find_pose_from_tag(K, res)
                rot, jaco = cv2.Rodrigues(pose[1], pose[1])
                # print(rot)
                print (pose)
                pts = res.corners.reshape((-1, 1, 2)).astype(np.int32)
                img = cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=5)
                cv2.circle(img, tuple(res.center.astype(np.int32)), 5, (0, 0, 255), -1)


                # our code
                x_backup = 0.25

                cam_w = 640
                cam_h = 360
                tol = 20

                z_speed_ub = 50
                z_speed_lb = 30
                p_factor = (z_speed_ub-z_speed_lb)/(cam_w/2 - tol)
                p = z_speed_lb + p_factor*(-tol)

                if res.center[0] > cam_w/2 + tol:
                    z_speed = p + p_factor*(res.center[0]-cam_w/2)
                elif res.center[0] < cam_w/2 - tol:
                    z_speed = -(p - p_factor*(res.center[0]-cam_w/2))
                else:
                    z_speed = 0

                side_length = abs(res.corners[0][0] - res.corners[2][0])

                x_speed_ub = 0.6
                x_speed_lb = 0.2
                q_factor = (x_speed_ub-x_speed_lb)/(40)
                q = x_speed_lb + q_factor*(-tol)


                if side_length > 80:
                    x_speed = -x_backup
                elif side_length < 50 and side_length > 10:
                    x_speed = -side_length*0.01 + 0.7
                elif side_length < 10:
                    x_speed = x_speed_ub
                else:
                    x_speed = 0

                # ep_chassis.drive_speed(x = x_speed, y = 0, z = z_speed)






            cv2.imshow("img", img)
            cv2.waitKey(10)

        except KeyboardInterrupt:
            ep_camera.stop_video_stream()
            ep_robot.close()
            ep_chassis.drive_speed(x = 0, y = 0, z = 0)
            print ('Exiting')
            exit(1)


