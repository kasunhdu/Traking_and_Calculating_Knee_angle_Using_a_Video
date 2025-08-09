
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = 'Snatch1.mp4'
cap = cv2.VideoCapture(video_path)


fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30.0

time_points = []
angle_values = []
frame_count = 0


with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

 
    def calAngle(a, b, c):
        a = np.array(a)  
        b = np.array(b)  
        c = np.array(c)  
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        ang = np.abs(radians * 180.0 / np.pi)
        if ang > 180.0:
            ang = 360 - ang
        return ang

    while True:
        ret, frame = cap.read() 


        rgb_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_im.flags.writeable = False

        result = pose.process(rgb_im)

        frame_count += 1
        current_time = frame_count / fps

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                R_h = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                R_k = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                R_a = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                angle = calAngle(R_h, R_k, R_a)

                time_points.append(current_time)
                angle_values.append(angle)

                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(255,0,255),thickness=2,circle_radius=2))

                h, w, c = frame.shape
                cv2.putText(frame, f'Angle: {int(angle)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            except IndexError:
                print(f"Skipping frame {frame_count}: Could not extract all required landmarks for angle calculation.")
            except Exception as e: 
                print(f"An error occurred processing landmarks in frame {frame_count}: {e}")
        else:
            print(f"No pose landmarks detected in frame {frame_count}.")

        cv2.imshow("PoseEstimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

if time_points and angle_values:
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, angle_values, marker='o', linestyle='-')
    plt.title('Knee Angle vs. Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    plt.show()
else:
    print("No angle data was collected to plot.")    