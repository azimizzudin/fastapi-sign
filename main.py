from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import pickle
import cv2
import mediapipe as mp
import numpy as np

app = FastAPI()

# Initialize Model Right
model_dict_right = pickle.load(open('./model_v4_right.p', 'rb'))
model_right = model_dict_right['model']

# Initialize Model Left
model_dict_left = pickle.load(open('./model_v4_left.p', 'rb'))
model_left = model_dict_left['model']

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize Label
labels_dict = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z', 'del': 'del', 'space': 'space', 'nothing': 'nothing'}

@app.get("/")
def example():
    return {"Example": "FastAPI"}
    
@app.post("/detect_fingerspelling_right")
async def detect_fingerspelling_right(file: UploadFile = File(...)):
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

                if len(data_aux) < 84:
                    data_aux.extend([0] * (84 - len(data_aux)))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model_right.predict([np.asarray(data_aux)])
            predictionStr = labels_dict[str(prediction[0])]
            
            return JSONResponse(content = {"message": predictionStr})
        
        else:
            return JSONResponse(content = {"message": " "})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
    
@app.post("/detect_fingerspelling_left")
async def detect_fingerspelling_left(file: UploadFile = File(...)):
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

                if len(data_aux) < 84:
                    data_aux.extend([0] * (84 - len(data_aux)))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model_left.predict([np.asarray(data_aux)])
            predictionStr = labels_dict[str(prediction[0])]
            
            return JSONResponse(content = {"message": predictionStr})
        
        else:
            return JSONResponse(content = {"message": " "})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
    
@app.post("/detect_sign_language_multiImage")
async def detect_sign_language(files: List[UploadFile] = File(...)):
    try:
        sentence = []

        for file in files:

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

                prediction = model_right.predict([np.asarray(data_aux)])
                predictionStr = labels_dict[str(prediction[0])]
                sentence.append(predictionStr)

            else:
                sentence.append(' ')
            
        return JSONResponse(content = {"message": ''.join(sentence)})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
