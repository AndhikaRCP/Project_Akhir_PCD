import numpy as np
import cv2
from keras.models import load_model
from PIL import Image, ImageOps

np.set_printoptions(suppress=True)
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def image_resizing(newWidth,image):
    print('INI IMAGE SHAPE' ,image.shape) 
    skalaFaktor = newWidth / image.shape[1] 
    print(skalaFaktor)
    newDimensi = (newWidth, int(image.shape[0] * skalaFaktor)) 
    resizingimage = cv2.resize(image, newDimensi, interpolation = cv2.INTER_AREA)
    return resizingimage

def image_preprocessing(inputImage):
    grayscale = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY) 
    clahe = cv2.createCLAHE(clipLimit = 6.0, tileGridSize = (8, 8))
    gray = clahe.apply(grayscale)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) 
    return blurred
class Enum(tuple):
        __getattr__ = tuple.index 

def coin_recog() :
    inputImage = cv2.imread("uploads/input_image.jpg") 
    inputImage = image_resizing(1024,inputImage)
    output = inputImage.copy() 
    imageAfterPreprocessing = image_preprocessing(inputImage)
    circles = cv2.HoughCircles(imageAfterPreprocessing, cv2.HOUGH_GRADIENT, dp = 2.2, minDist = 100, param1 = 200, param2 = 100, minRadius = 50, maxRadius = 120)
    Material = Enum(('seratus', 'dua_ratus', 'lima_ratus', 'seribu'))

    def predictMaterial(roi):
        size = (224, 224)
        roi_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        image = ImageOps.fit(roi_image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        
        if class_name in Material:
            return class_name
        else:
            raise ValueError(f"Unknown class name: {class_name}")

    diameter = []
    materials = []
    coordinates = []
    count = 0
  
    if circles is not None:
        for (koordinat_x, koordinat_y, radius) in circles[0, :]:
            diameter.append(radius)

        circles = np.round(circles[0, :]).astype("int") 

        for (koordinat_x, koordinat_y, radius) in circles: 
            count += 1
            coordinates.append((koordinat_x, koordinat_y)) 
            roi = inputImage[koordinat_y - radius:koordinat_y + radius, koordinat_x - radius:koordinat_x + radius] # extract region of interest
            material = predictMaterial(roi)
            materials.append(material) 
            cv2.circle(output, (koordinat_x, koordinat_y), radius, (0, 255, 0), 2)
            cv2.putText(output, material, (koordinat_x - 40, koordinat_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness = 2, lineType = cv2.LINE_AA)

    i = 0
    nominalUang = 0
    countCoinBasedCategory = {
        'koinSeratus': 0,
        'koinDuaratus': 0,
        'koinLimaratus': 0,
        'koinSeribu': 0,
        }
    
    while i < len(diameter): 
        jenis = materials[i] 
        (koordinat_x, koordinat_y) = coordinates[i] 
        if jenis == "lima_ratus": 
            text = "500"
            countCoinBasedCategory["koinLimaratus"] += 1
            nominalUang += 500
        elif jenis == "seribu": 
            text = "1.000"
            countCoinBasedCategory["koinSeribu"] += 1
            nominalUang += 1000
        elif jenis == "seratus": 
            text = "100"
            countCoinBasedCategory["koinSeratus"] += 1
            nominalUang += 100
        elif jenis == "dua_ratus": 
            text = "200"
            countCoinBasedCategory["koinDuaratus"] += 1
            nominalUang += 200
        else: 
            text = "Unknown"
        cv2.putText(output, text, (koordinat_x - 40, koordinat_y + 22), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness = 2, lineType = cv2.LINE_AA)
        i += 1

    output = image_resizing(800,output)
    cv2.imwrite("uploads/output_image.jpg", output)
    
    dataOutput = {
        "jumlahKoin" : count,
        "jumlahKoinBasedCategory" : countCoinBasedCategory,
        "nominalUang" : nominalUang
    }

    return dataOutput