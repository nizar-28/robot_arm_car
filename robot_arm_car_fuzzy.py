# Code references from CrazyCurly and ZainabIshtiaq2001
# Links: https://github.com/Crazycurly/gesture_MeArm/blob/main/python/main.py,
#        https://github.com/ZainabIshtiaq2001/Hand_Gesture_Car_Control/blob/main/mediapipscode.py

import serial
import cv2
import mediapipe as mp
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Configuration
write_video = True
debug = False

if not debug:
    ser = serial.Serial('COM9', 115200)

x_min = 0
x_mid = 90
x_max = 180

# Use x-axis distance between wrist and index finger MCP to control x axis
palm_angle_min = -50
palm_angle_mid = 20

z_min = 0
z_mid = 135
z_max = 180
# Use palm size to control z axis
plam_size_min = 0.1
plam_size_max = 0.3

claw_open_angle = 90
claw_close_angle = 180

servo_angle = [x_mid,z_mid,claw_open_angle] # [x, z, claw]
prev_servo_angle = servo_angle
fist_threshold = 7

# Fuzzy Logic setup for Servo X
angle_diff = ctrl.Antecedent(np.arange(-180, 181, 1), 'angle_diff')
angle_diff['negative'] = fuzz.trimf(angle_diff.universe, [-180, -90, 0])
angle_diff['zero'] = fuzz.trimf(angle_diff.universe, [-90, 0, 90])
angle_diff['positive'] = fuzz.trimf(angle_diff.universe, [0, 90, 180])

servo_x_ctrl = ctrl.Consequent(np.arange(x_min, x_max + 1, 1), 'servo_x')
servo_x_ctrl['low'] = fuzz.trimf(servo_x_ctrl.universe, [x_min, x_mid, x_mid])
servo_x_ctrl['medium'] = fuzz.trimf(servo_x_ctrl.universe, [x_mid, x_mid, x_max])

rule1 = ctrl.Rule(angle_diff['negative'], servo_x_ctrl['low'])
rule2 = ctrl.Rule(angle_diff['zero'], servo_x_ctrl['medium'])
rule3 = ctrl.Rule(angle_diff['positive'], servo_x_ctrl['medium'])

servo_x_ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3])
servo_x_ctrl_simulation = ctrl.ControlSystemSimulation(servo_x_ctrl_system)

# Fuzzy Logic setup for Servo Z
palm_size_diff = ctrl.Antecedent(np.arange(plam_size_min, plam_size_max + 0.01, 0.01), 'palm_size_diff')
palm_size_diff['small'] = fuzz.trimf(palm_size_diff.universe, [plam_size_min, plam_size_min, (plam_size_min + plam_size_max) / 2])
palm_size_diff['medium'] = fuzz.trimf(palm_size_diff.universe, [plam_size_min, (plam_size_min + plam_size_max) / 2, plam_size_max])
palm_size_diff['large'] = fuzz.trimf(palm_size_diff.universe, [(plam_size_min + plam_size_max) / 2, plam_size_max, plam_size_max])

servo_z_ctrl = ctrl.Consequent(np.arange(z_min, z_max + 1, 1), 'servo_z')
servo_z_ctrl['low'] = fuzz.trimf(servo_z_ctrl.universe, [z_min, z_mid, z_mid])
servo_z_ctrl['medium'] = fuzz.trimf(servo_z_ctrl.universe, [z_mid, z_mid, z_max])

rule4 = ctrl.Rule(palm_size_diff['small'], servo_z_ctrl['low'])
rule5 = ctrl.Rule(palm_size_diff['medium'], servo_z_ctrl['medium'])
rule6 = ctrl.Rule(palm_size_diff['large'], servo_z_ctrl['medium'])

servo_z_ctrl_system = ctrl.ControlSystem([rule4, rule5, rule6])
servo_z_ctrl_simulation = ctrl.ControlSystemSimulation(servo_z_ctrl_system)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

fingertip_keypoints=[4,8,12,16,20]
total = 0

cap = cv2.VideoCapture(0)

# video writer
if write_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
map_range = lambda x, in_min, in_max, out_min, out_max: abs((x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min)

# Check if it is fist
def is_fist(hand_landmarks, palm_size):
    # calculate the distance between the wrist and the each finger tips divided by palm size and compare with fist_threshold
    distance_sum = 0
    WRIST = hand_landmarks.landmark[0]
    for i in [7,8,11,12,15,16,19,20]:
        distance_sum += ((WRIST.x - hand_landmarks.landmark[i].x)**2 + \
                         (WRIST.y - hand_landmarks.landmark[i].y)**2 + \
                         (WRIST.z - hand_landmarks.landmark[i].z)**2)**0.5
    return distance_sum/palm_size < fist_threshold

def landmark_to_servo_angle(hand_landmarks):
    servo_angle = [x_mid,z_mid,claw_open_angle]
    WRIST = hand_landmarks.landmark[0]
    INDEX_FINGER_MCP = hand_landmarks.landmark[5]
    # Calculate the distance between the wrist and the index finger MCP
    palm_size = ((WRIST.x - INDEX_FINGER_MCP.x)**2 + (WRIST.y - INDEX_FINGER_MCP.y)**2 + (WRIST.z - INDEX_FINGER_MCP.z)**2)**0.5

    if is_fist(hand_landmarks, palm_size):
        servo_angle[2] = claw_close_angle
    else:
        servo_angle[2] = claw_open_angle

    # Calculate x-axis distance between wrist and index finger MCP
    distance = palm_size
    angle = (WRIST.x - INDEX_FINGER_MCP.x) / distance  # calculate the radian between the wrist and the index finger
    angle = int(angle * 180 / 3.1415926)               # convert radian to degree
    angle = clamp(angle, palm_angle_min, palm_angle_mid)
    servo_angle[0] = map_range(angle, palm_angle_min, palm_angle_mid, x_max, x_min)


    # Use palm size as z angle
    palm_size = clamp(palm_size, plam_size_min, plam_size_max)
    servo_angle[1] = map_range(palm_size, plam_size_min, plam_size_max, z_max, z_min)

    # float to int
    servo_angle = [int(i) for i in servo_angle]

    return servo_angle

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Please ignore empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        landmark_list = []

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:            
                if len(results.multi_hand_landmarks) == 1:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    # Landmark to servo angle calculation
                    servo_angle = landmark_to_servo_angle(hand_landmarks)

                    # Fuzzy Logic Simulation for Servo X
                    angle_difference = servo_angle[0] - prev_servo_angle[0]
                    servo_x_ctrl_simulation.input['angle_diff'] = angle_difference
                    servo_x_ctrl_simulation.compute()
                    servo_x_angle = servo_x_ctrl_simulation.output['servo_x']

                    # Fuzzy Logic Simulation for Servo Z
                    palm_size_difference = servo_angle[1] - prev_servo_angle[1]
                    servo_z_ctrl_simulation.input['palm_size_diff'] = palm_size_difference
                    servo_z_ctrl_simulation.compute()
                    servo_z_angle = servo_z_ctrl_simulation.output['servo_z']

                    # Update prev_servo_angle for next iteration
                    prev_servo_angle = servo_angle
                    if not debug:
                        ser.write(bytearray(servo_angle))
                
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h,w,c=image.shape
                        cx,cy= int(lm.x*w), int(lm.y*h)
                        landmark_list.append([id,cx,cy])               
            
                                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
        fingers = []
        threshold = 1.1
        if len(landmark_list) != 0:
            # Check the distance from the wrist to each fingertip
            for id in range(0, 5):
                wrist_x, wrist_y = landmark_list[0][1], landmark_list[0][2]
                palm = ((wrist_x - landmark_list[5][1])**2 + (wrist_y - landmark_list[5][2])**2)**0.5
                fingertip_x, fingertip_y = landmark_list[fingertip_keypoints[id]][1], landmark_list[fingertip_keypoints[id]][2]
                
                # Calculate the Euclidean distance
                distance = ((wrist_x - fingertip_x) ** 2 + (wrist_y - fingertip_y) ** 2) ** 0.5 / palm
                
                # Use a threshold to determine if the finger is extended or not
                if distance > threshold:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total = fingers.count(1)
            print(fingers)

            if not debug:
                print(total)
                ser.write(str(total).encode())
         
        # Image is flipped horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        # show servo angle
        cv2.putText(image, str(servo_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if total==0 or total==5:
            
            cv2.putText(image, "Stop", (10, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 0, 255), 5)
            print("Stop")
            

        elif total==1:

            cv2.putText(image, "Forward", (10, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 5)
            print("Forward")

        elif total==2:

            cv2.putText(image, "Right", (10, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 5)
            print("Right")


        elif total==3:
            cv2.putText(image, "Left", (10, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 5)
            print("Left")

        elif total==4:
            cv2.putText(image, "Backward", (10, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 5)
            print("Backward")

        cv2.imshow('MediaPipe Hands', image)

        if write_video:
            out.write(image)
        if cv2.waitKey(5) & 0xFF == 27:
            if write_video:
                out.release()
            break
cap.release()
