import cv2
import mediapipe as mp
import numpy as np
import socket
from face_geometry import get_metric_landmarks, PCF, procrustes_landmark_basis

# %% 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

mp_face_mesh = mp.solutions.face_mesh


# %% PPre-config
window_name = 'MediaPipe Hands'
cap = cv2.VideoCapture(0)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 120)


# pseudo camera internals
focal_length = frame_width
center = (frame_width/2, frame_height/2)
camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double"
                        )
dist_coeff = np.zeros((4, 1))

pcf = PCF(near=1,far=10000,frame_height=frame_height,frame_width=frame_width,fy=camera_matrix[1,1])

points_idx =  [33,263,61,291,199] # [k for k in range(0,468)] 
points_idx = points_idx + [key for (key,val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()
wrist = []

# %% Send data to server
def send_to_server(hand_state, coords):
    # try:
      azim = coords.item(0)*180 - 90
      print(azim)
      txt = '{state}, {azimuth}'.format(state=hand_state, 
                                        azimuth=azim)
      s.sendto(txt.encode(), (IP,PORT)) #send message back
    # except:
    #   print('Sending UDP failed!')

# Initialize UDP server
global s, IP, PORT
IP = '127.0.0.1'  # Symbolic name meaning all available interfaces
PORT = 50060      # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 

# %% Calculate finger angles

def get_finger_angle(a,b,c):
  angle = get_triangle_angle(a, b, c)
  # Return state
  if angle <= 90:
    # print('hand is closed')
    return 1
  else:     
    # print('handd is open')
    return 0   

def get_triangle_angle(a, b, c):
    # a: wrist coordinates
    # b: base of the finger coordinates
    # c: tip of thee finger coordinates   
    ba = a - c
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle
    
def landmark2numpy(landmark):
    h = np.array([landmark.x, landmark.y, landmark.y])
    return h
  
     
# %% RUN MODEL       
with mp_hands.Hands( model_complexity=0, min_detection_confidence=0.5,
                     min_tracking_confidence=0.5) as hands, \
     mp_face_mesh.FaceMesh( min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    face_results = face_mesh.process(image)
    
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
      # Check if hand is open or closed 
      wrist = landmark2numpy(hand_landmarks.landmark[0])  
      finger_base = landmark2numpy(hand_landmarks.landmark[9])
      finger_tip = landmark2numpy(hand_landmarks.landmark[12])  
      hand_state = get_finger_angle(wrist, finger_base, finger_tip)
      
      # UDP Listening to ports
      coords = send_to_server(hand_state, wrist) 
    
    
    ######### FACE TRACKING 
    if face_results.multi_face_landmarks:
      face_landmarks = face_results.multi_face_landmarks[0]
      landmarks = np.array([(lm.x,lm.y,lm.z) for lm in face_landmarks.landmark])
      landmarks = landmarks.T
      
      metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)
      model_points = metric_landmarks[0:3, points_idx].T
      image_points = landmarks[0:2, points_idx].T * np.array([frame_width, frame_height])[None,:]

      success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)

      (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 15.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeff)
      
      L_ear = 132 # coordinate index
      R_ear = 361 # coordinate index
      for ii in (L_ear, R_ear):
        pos = np.array((frame_width*landmarks[0, ii], frame_height*landmarks[1, ii])).astype(np.int32)
        image = cv2.circle(image, tuple(pos), 2, (0, 255, 0), -1)
      
    if wrist != []:
      hand_angle = get_triangle_angle(landmarks[:, L_ear],landmarks[:,R_ear], wrist)
      wrist = []
      
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    
    # Bring screen to the front 
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    # Kill the process by pressing 'Esc' or ressing 'quit'
    if cv2.waitKey(5) & 0xFF == 27:
      break    
    if cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE) < 1: 
      break
            
# print('Goodbye!')      
cv2.destroyAllWindows()
cap.release()



# %%

# %%

# %%






    
    
    
    
    
    
    
    
    
    
    
    
# %%

# %%

# %%
