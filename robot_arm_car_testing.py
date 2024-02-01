# Code references from CrazyCurly and ZainabIshtiaq2001
# Links: https://github.com/Crazycurly/gesture_MeArm/blob/main/python/main.py,
#        https://github.com/ZainabIshtiaq2001/Hand_Gesture_Car_Control/blob/main/mediapipscode.py

import serial
import cv2
import mediapipe as mp

# Configuration
write_video = True
debug = False

if not debug:
    ser = serial.Serial('COM9', 115200) #COM9 for wire

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

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        # Proportional term
        P = self.kp * error

        # Integral term
        self.integral += error
        I = self.ki * self.integral

        # Derivative term
        D = self.kd * (error - self.prev_error)
        self.prev_error = error

        # Output value
        output = P + I + D
        return output

# Initialize PID controllers for x-axis and z-axis
pid_x = PIDController(kp=2, ki=0.01, kd=0.01)
pid_z = PIDController(kp=0.5, ki=0.01, kd=0.01)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

fingertip_keypoints=[4,8,12,16,20]
total = 0

# Low-pass filter parameters
alpha_x = 0.1  # You can adjust this value to control the smoothing effect for servo x
alpha_z = 0.2 # Smoothing effect for servo z

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
    global servo_angle, prev_servo_angle
    servo_angle = [x_mid, z_mid, claw_open_angle]
    WRIST = hand_landmarks.landmark[0]
    INDEX_FINGER_MCP = hand_landmarks.landmark[5]
    # Calculate the distance between the wrist and the index finger MCP
    palm_size = ((WRIST.x - INDEX_FINGER_MCP.x) ** 2 + (WRIST.y - INDEX_FINGER_MCP.y) ** 2 + (
                WRIST.z - INDEX_FINGER_MCP.z) ** 2) ** 0.5

    if not is_fist(hand_landmarks, palm_size):
        servo_angle[2] = claw_close_angle
    else:
        servo_angle[2] = claw_open_angle

    # Calculate x-axis distance between wrist and index finger MCP
    distance = palm_size
    angle = (WRIST.x - INDEX_FINGER_MCP.x) / distance  # calculate the radian between the wrist and the index finger
    angle = int(angle * 180 / 3.1415926)  # convert radian to degree
    angle = clamp(angle, palm_angle_min, palm_angle_mid)
    servo_angle[0] = map_range(angle, palm_angle_min, palm_angle_mid, x_max, x_min)

    # Use palm size as z angle
    palm_size = clamp(palm_size, plam_size_min, plam_size_max)
    servo_angle[1] = map_range(palm_size, plam_size_min, plam_size_max, z_max, z_min)

    # Calculate error for x-axis (servo[0])
    error_x = servo_angle[0] - prev_servo_angle[0]
    # Update PID controller for x-axis
    pid_output_x = pid_x.update(error_x)
    # Adjust servo[0] angle
    servo_angle[0] += int(pid_output_x)

    servo_angle[0] = alpha_x * pid_output_x + (1 - alpha_x) * prev_servo_angle[0]

    servo_angle[0] = clamp(servo_angle[0], 0, 180)

    # Calculate error for z-axis (servo[1])
    error_z = servo_angle[1] - prev_servo_angle[1]
    # Update PID controller for z-axis
    pid_output_z = pid_z.update(error_z)
    # Adjust servo[1] angle
    servo_angle[1] += int(pid_output_z)

    servo_angle[1] = alpha_z * pid_output_z + (1 - alpha_z) * prev_servo_angle[1]

    servo_angle[1] = clamp(servo_angle[1], 0, 180)

    # Store current servo angles for the next iteration
    prev_servo_angle = servo_angle.copy()

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
                    servo_angle = landmark_to_servo_angle(hand_landmarks)

                    current_servo_angle_x = servo_angle[0]
                    current_servo_angle_z = servo_angle[1]

                    # Dynamically adjust desired angles based on current servo positions
                    desired_angle_x = current_servo_angle_x
                    desired_angle_z = current_servo_angle_z

                    if servo_angle != prev_servo_angle:
                        print("Servo angle: ", servo_angle)
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
        thumb_threshold = 1.6
        fourfingers_threshold = 1.3
        if len(landmark_list) != 0:
            # Check the distance from the wrist to each fingertip
            wrist_x, wrist_y = landmark_list[0][1], landmark_list[0][2]
            palm = ((wrist_x - landmark_list[2][1])**2 + (wrist_y - landmark_list[2][2])**2)**0.5
            palm_fourfingers = ((wrist_x - landmark_list[5][1])**2 + (wrist_y - landmark_list[5][2])**2)**0.5
            thumbtip_x, thumbtip_y = landmark_list[fingertip_keypoints[0]][1], landmark_list[fingertip_keypoints[0]][2]

            # Calculate the Euclidean distance
            distance_thumb = ((wrist_x - thumbtip_x)**2 + (wrist_y - thumbtip_y)**2) **0.5 / palm

            # Use a threshold to determine if thumb is extended or not
            if distance_thumb > thumb_threshold:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                fingertip_x, fingertip_y = landmark_list[fingertip_keypoints[id]][1], landmark_list[fingertip_keypoints[id]][2]
                
                distance_fourfingers = ((wrist_x - fingertip_x) ** 2 + (wrist_y - fingertip_y) ** 2) ** 0.5 / palm_fourfingers
                
                # Use a threshold to determine if the finger is extended or not
                if distance_fourfingers > fourfingers_threshold:
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
