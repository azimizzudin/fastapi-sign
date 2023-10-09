from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pickle
import cv2
import mediapipe as mp
import numpy as np

app = FastAPI()

# Initialize Model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

@app.get("/")
def first_example():
    return {"GFG Example": "FastAPI"}
    
@app.post("/detect_sign_language")
async def detect_sign_language(file: UploadFile):
    try:
        data_aux = []
        x_ = []
        y_ = []

        frame = cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]
            
            return JSONResponse(content = {"char": predicted_character})
        
        else:
            return JSONResponse(content = {"message": "No hand detected in the frame."})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)})