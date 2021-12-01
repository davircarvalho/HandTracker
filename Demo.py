import cv2
import mediapipe as mp
import numpy as np
import socket


# %% 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# # For static images:
# IMAGE_FILES = []
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # Read an image, flip it around y-axis for correct handedness output (see
#     # above).
#     image = cv2.flip(cv2.imread(file), 1)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Print handedness and draw hand landmarks on the image.
#     print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       print(
#           f'Index finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#       )
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
#     cv2.imwrite(
#         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
#     # Draw hand world landmarks.
#     if not results.multi_hand_world_landmarks:
#       continue
#     for hand_world_landmarks in results.multi_hand_world_landmarks:
#       mp_drawing.plot_landmarks(
#         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)





# %% PPre-config
window_name = 'MediaPipe Hands'
cap = cv2.VideoCapture(0)
image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 120)


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
def get_finger_angle(a, b, c):
    # a: wrist coordinates
    # b: base of the finger coordinates
    # c: tip of thee finger coordinates   
    ba = a - c
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    # Return state
    if angle <= 90:
      print('hand is closed')
      return 1
    else:     
      print('handd is open')
      return 0
    
def landmark2numpy(landmark):
    h = np.array([landmark.x, landmark.y, landmark.y])
    return h
  
     
# %% RUN MODEL       
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
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
    
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    
    # Bring screen to the front 
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    # Kill the process by pressing 'Esc'
    if cv2.waitKey(5) & 0xFF == 27:
      break
    
    
    # if cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE) < 1: 
    #           break
print('Goodbye!')      
cv2.destroyAllWindows()
cap.release()



# %%






    
    
    
    
    
    
    
    
    
    
    
    
# %%

# %%

# %%
