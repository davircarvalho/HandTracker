
#%% 
import cv2
import mediapipe as mp
import numpy as np
import socket
from face_geometry import get_metric_landmarks, PCF, procrustes_landmark_basis
from cheapFilter import OneEuroFilter

# %% 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# %% Filter configuration
min_cutoff = 0.004
beta = 0.7
cont = 1 
one_euro_filter = OneEuroFilter(0, 0,
                               min_cutoff=min_cutoff,
                               beta=beta)

# %% PPre-config
window_name = 'MediaPipe Hands'
cap = cv2.VideoCapture(0)
# Bring window to the front 
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# cap.set(cv2.CAP_PROP_FPS, 60)
blank_image = np.zeros(shape= (int(frame_height), int(frame_width)))
cv2.imshow(window_name,blank_image )
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)




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
hand_success = False
head_success = False
hnd_coords = []


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

def calc_points_dist(a, b):
    x1 = a.item(0)
    y1 = a.item(1)
    z1 = a.item(2)
    x2 = b.item(0)
    y2 = b.item(1)
    z2 = b.item(2)
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    return (dist/100)

def get_head_orientation():
    rvec = face_rotation_vector
    tvec = face_translation_vector
    rmat = cv2.Rodrigues(rvec)[0]  
    P = np.hstack((rmat,tvec)) # projection matrix
    
    # find euler angles 
    euler_angles =  cv2.decomposeProjectionMatrix(P)[6]
    pitch = euler_angles.item(0) 
    yaw = euler_angles.item(1) 
    roll = euler_angles.item(2) 

    # Ajust coordinate ranges
    if pitch < 0:
      pitch = 180 + pitch
    else:
      pitch = pitch - 180

    orientation = [round(yaw), round(pitch), round(roll)] 
    return orientation


# %% Send data to server
def send_to_server(hnd_state, hnd_coords):
    try:
      head_orientation = get_head_orientation()
      txt = '{head_yaw},{head_pitch}, {head_roll},{hand_state}, {hand_azim}, {hand_elev}, {hand_radius}'.format(
              head_yaw = head_orientation[0],
              head_pitch = head_orientation[1],
              head_roll = head_orientation[2],
              hand_state = hnd_state, 
              hand_azim = hnd_coords[0],
              hand_elev = hnd_coords[1],
              hand_radius = hnd_coords[2])
      s.sendto(txt.encode(), (IP,PORT)) #send message back
    except:
      print('Sending UDP failed!')

# Initialize UDP server
global s, IP, PORT
IP = '127.0.0.1'  # Symbolic name meaning all available interfaces
PORT = 50050      # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 


#########################################################################################
# %%######################### TRACKING MODELS ###########################################  
global rotation_vector, translation_vector, hand_coords
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
    hand_results = hands.process(image)
    face_results = face_mesh.process(image)
    
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
# %% HAND TRACKING ###################################################################
    hand_translation_vector = np.empty(shape=(0))
    if hand_results.multi_hand_landmarks:
      # Draw hand landmarks
      for hand_landmarks in hand_results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)

      h_model_landmarks = hand_results.multi_hand_world_landmarks[0]
      hand_landmarks = hand_results.multi_hand_landmarks[0]
      h_model_points = []
      h_landmarks = []
      
      dx = [0,5,17, 18]
      for ii, (lw, lm) in enumerate(zip(h_model_landmarks.landmark, hand_landmarks.landmark)):
        if ii in dx:
          h_model_points.append([lw.x,lw.y,lw.z])
          h_landmarks.append([lm.x,lm.y,lm.z])
      h_model_points = np.array(h_model_points)
      h_landmarks = np.array(h_landmarks)
      
      h_image_points = h_landmarks[:,0:2] * np.array([frame_width, frame_height])[None,:]
      _, hand_rotation_vector, hand_translation_vector = cv2.solvePnP(h_model_points, 
                                                                            h_image_points, 
                                                                            camera_matrix, 
                                                                            dist_coeff, 
                                                                            flags=cv2.SOLVEPNP_P3P)      
    if hand_translation_vector is not None and hand_translation_vector.any():
      hand_success = True
    else:
      hand_success = False
    
    
# %% FACE TRACKING ###################################################
    if face_results.multi_face_landmarks:
      face_landmarks = face_results.multi_face_landmarks[0]
      landmarks = np.array([(lm.x,lm.y,lm.z) for lm in face_landmarks.landmark])
      landmarks = landmarks.T
      
      image_points = landmarks[0:2, points_idx].T * np.array([frame_width, frame_height])[None,:]
      
      metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)
      model_points = metric_landmarks[0:3, points_idx].T
      
      _, face_rotation_vector, face_translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)
      (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 15.0)]), face_rotation_vector, face_translation_vector, camera_matrix, dist_coeff)
      
      L_ear = 132 # coordinate index
      R_ear = 361 # coordinate index
      for ii in (L_ear, R_ear):
        pos = np.array((frame_width*landmarks[0, ii], frame_height*landmarks[1, ii])).astype(np.int32)
        image = cv2.circle(image, tuple(pos), 2, (0, 255, 0), -1)
      head_success = True
    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    
    
######################### ID GESTURES AND RELATIVE POSITION ############################################################
# %% Detect specific hand gestures
    if hand_success:      
        hand_translation_vector = hand_translation_vector*100
        hand_translation_vector[2] = abs(hand_translation_vector[2])
    
        # Check if hand is open or closed 
        wrist = landmark2numpy(hand_landmarks.landmark[0])  
        finger_base = landmark2numpy(hand_landmarks.landmark[9])
        finger_tip = landmark2numpy(hand_landmarks.landmark[12])  
        hand_state = get_finger_angle(wrist, finger_base, finger_tip)
    
    if head_success and not hand_success:
      hand_state = 0 # defaults to open hands      
      if len(hnd_coords) < 3: # case there are no hand coords previously calculated (in another iteraction)
        hnd_coords = [0,0,0]
      send_to_server(hand_state, hnd_coords) 
    
    # Calculate angles between hands and ears  
    if hand_success and head_success: # if hands and face are sucessful
      face_normal = face_translation_vector.copy()
      face_normal[2] = -1 # negative to ensure the point is always behind the screen
      hand_azimuth = get_triangle_angle(face_normal.flatten(),
                                       face_translation_vector.flatten(),                                       
                                       hand_translation_vector.flatten())
      ####################### DATA OUTPUT ####################################################
      if hand_translation_vector[0] < face_translation_vector[0]:
        hand_azimuth = -hand_azimuth
      hand_elevation = 0
      hand_radius = calc_points_dist(face_translation_vector, hand_translation_vector)
      hnd_coords = [hand_azimuth, hand_elevation,  hand_radius]
      # Filter       
      hnd_coords[0] = one_euro_filter(cont, hnd_coords[0])
      cont +=1
      
      # UDP courrier
      send_to_server(hand_state, hnd_coords) 
        
      # Image: Write translation vectors (top left) 
      round_factor = 0
      txt = '{x}, {y}, {z}'.format(x = f"{round(hand_translation_vector.item(0), round_factor):+.1f}",
                                  y = f"{round(hand_translation_vector.item(1), round_factor):+.1f}",
                                  z = f"{round(hand_translation_vector.item(2), round_factor):+.1f}")
      image = cv2.putText(image, txt, (00, 20  ), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (230, 80, 80), 2, cv2.LINE_AA)   
      txt = '{x}, {y}, {z}'.format(x = f"{round(face_translation_vector.item(0), round_factor):+.1f}",
                                  y = f"{round(face_translation_vector.item(1), round_factor):+.1f}",
                                  z = f"{round(face_translation_vector.item(2), round_factor):+.1f}")
      image = cv2.putText(image, txt, (00, 40  ), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 140, 20), 2, cv2.LINE_AA)   
      txt = '{x}, {y}, {z}'.format(x = f"{hnd_coords[0]:+.1f}",
                                  y = f"{hnd_coords[1]:+.1f}",
                                  z = f"{hnd_coords[2]:+.2f}")
      image = cv2.putText(image, txt, (00, 60  ), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (140, 0, 120), 2, cv2.LINE_AA)   
       
    # Show image
    cv2.imshow(window_name, image)
    
    # Kill the process by pressing 'Esc' or ressing 'quit'
    if cv2.waitKey(5) & 0xFF == 27:
      break    
    if cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE) < 1: 
      break         
# print('Goodbye!')      
cv2.destroyAllWindows()
cap.release()

