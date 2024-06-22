import numpy as np
import glob
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def image_resizing(newWidth,image):
    # resize image while retaining aspect ratio
    print('INI IMAGE SHAPE' ,image.shape) 
    skalaFaktor = newWidth / image.shape[1] #mengambil width dari gambar input
    print(skalaFaktor)
    newDimensi = (newWidth, int(image.shape[0] * skalaFaktor)) #<--- tuple berisi dimensi (width,height)
    resizingimage = cv2.resize(image, newDimensi, interpolation = cv2.INTER_AREA)
    return resizingimage

def image_preprocessing(inputImage):
    grayscale = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY) # convert image to grayscale
    # create a CLAHE object to apply contrast limiting adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit = 6.0, tileGridSize = (8, 8))
    gray = clahe.apply(grayscale)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) # blur image
    return blurred

def calc_histogram(img):
        m = np.zeros(img.shape[:2], dtype="uint8") # create mask
        (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
        cv2.circle(m, (w, h), 60, 255, -1) # draw circle
        # calcHist expects a list of images, color channels, mask, bins, ranges
        h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(h, h).flatten() # return normalized "flattened" histogram
    
# define Enum class
class Enum(tuple):
        __getattr__ = tuple.index # define __getattr__ to return index of tuple
        
def calc_hist_from_file(file):
        img = cv2.imread(file)
        return calc_histogram(img) # return histogram

def coin_recog() :
    inputImage = cv2.imread("uploads/input_image.jpg") # read in image
    inputImage = image_resizing(1024,inputImage)
    output = inputImage.copy() # create a copy of the image to display results
    imageAfterPreprocessing = image_preprocessing(inputImage)
    # apply edge detection to image
    circles = cv2.HoughCircles(imageAfterPreprocessing, cv2.HOUGH_GRADIENT, dp = 2.2, minDist = 100, param1 = 200, param2 = 100, minRadius = 50, maxRadius = 120)
    # Enumerate material types for use in classifier
    Material = Enum(('lima_ratus', 'seribu', 'seratus', 'dua_ratus'))

    # locate sample image files
    sample_images_lima_ratus = glob.glob("sample_images/lima_ratus/*.png")
    sample_images_seribu = glob.glob("sample_images/seribu/*.png")
    sample_images_seratus = glob.glob("sample_images/seratus/*.png")
    sample_images_dua_ratus = glob.glob("sample_images/dua_ratus/*.png")

    # define training data and labels
    DATAS = []
    labels = []

    # compute and store training data and labels
    for i in sample_images_lima_ratus:
        DATAS.append(calc_hist_from_file(i))
        labels.append(Material.lima_ratus)
    for i in sample_images_seribu:
        DATAS.append(calc_hist_from_file(i))
        labels.append(Material.seribu)
    for i in sample_images_seratus:
        DATAS.append(calc_hist_from_file(i))
        labels.append(Material.seratus)
    for i in sample_images_dua_ratus:
        DATAS.append(calc_hist_from_file(i))
        labels.append(Material.dua_ratus)

    # instantiate classifier
    classifier = MLPClassifier(solver = "adam", hidden_layer_sizes=(1000,), max_iter=100000,learning_rate="adaptive")

    # split samples into training and test data
    DATA_train, DATA_test, label_train, label_test = train_test_split(DATAS, labels, test_size = 0.5)

    # train and score classifier
    classifier.fit(DATA_train, label_train) # train classifier
    score = int(classifier.score(DATA_test, label_test) * 100) # calculate score
    print("Classifier mean accuracy: ", score) # print accuracy

    def predictMaterial(roi):
        hist = calc_histogram(roi) # calculate feature vector for region of interest
        s = classifier.predict([hist]) # predict material type
        return Material[int(s)] # return predicted material type

    diameter = []
    materials = []
    coordinates = []

    count = 0
    # loop over the detected circles
    if circles is not None:
        # append radius to list of diameters
        for (koordinat_x, koordinat_y, radius) in circles[0, :]:
            diameter.append(radius)

        circles = np.round(circles[0, :]).astype("int") # convert coordinates and radii to integers

        for (koordinat_x, koordinat_y, radius) in circles: # loop over coordinates and radii of the circles
            count += 1
            coordinates.append((koordinat_x, koordinat_y)) # add coordinates to list
            roi = inputImage[koordinat_y - radius:koordinat_y + radius, koordinat_x - radius:koordinat_x + radius] # extract region of interest
            material = predictMaterial(roi) # predict material type
            materials.append(material) # add material type to list
            # membuat lingkaran pada setiap gambar koin
            cv2.circle(output, (koordinat_x, koordinat_y), radius, (0, 255, 0), 2)
            
            # Teks nama jenis koin
            cv2.putText(output, material, (koordinat_x - 40, koordinat_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness = 2, lineType = cv2.LINE_AA)

    # get biggest diameter
    biggest = max(diameter)
    indexBiggestDiameter = diameter.index(biggest) # get index of biggest diameter
    print(diameter) # print diameters

    # scale everything according to maximum diameter
    if materials[indexBiggestDiameter] == "seribu":
        diameters = [x / biggest * 20.75 for x in diameter]
        scaledTo = "Skala berdsasarkan koin 1.000"
    elif materials[indexBiggestDiameter] == "lima_ratus":
        diameters = [x / biggest * 15.25 for x in diameter]
        scaledTo = "Skala berdsasarkan koin 500"
    elif materials[indexBiggestDiameter] == "seratus":
        diameters = [x / biggest * 18.25 for x in diameter]
        scaledTo = "Skala berdsasarkan koin 100"
    elif materials[indexBiggestDiameter] == "dua_ratus":
        diameters = [x / biggest * 20.25 for x in diameter]
        scaledTo = "Skala berdsasarkan koin 200"
    else:
        scaledTo = "unable to scale.."

#####################################################
    i = 0
    nominalUang = 0
    countCoinBasedCategory = {
        'koinSeratus': 0,
        'koinDuaratus': 0,
        'koinLimaratus': 0,
        'koinSeribu': 0,
        }
    
    while i < len(diameters): # loop over diameters
        # d = diameters[i] # get diameter
        # print('d = ',d)
        print('len d = ',len(diameters))
        print('i = ',i)
        jenis = materials[i] # get material type
        (koordinat_x, koordinat_y) = coordinates[i] # get coordinates
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
        # Teks tulisan nominal koin 
        cv2.putText(output, text, (koordinat_x - 40, koordinat_y + 22), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness = 2, lineType = cv2.LINE_AA)
        i += 1

    # resize output image while retaining aspect ratio
    output = image_resizing(800,output)

    # write summary on output image
    cv2.putText(output, scaledTo, (5, output.shape[0] - 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), lineType = cv2.LINE_AA)
    cv2.imwrite("uploads/output_image.jpg", output)
    # print('JUMLAH KOIN : ',count)
    # print('KATEGORI KOIN', countCoinBasedCategory)
    # print('Nominal KOIN', nominalUang)
    
    dataOutput = {
        "jumlahKoin" : count,
        "jumlahKoinBasedCategory" : countCoinBasedCategory,
        "nominalUang" : nominalUang
    }
    
    # print('jumlah Koin ', dataOutput["jumlahKoin"])
    # print('jumlah Koin ', dataOutput["jumlahKoinBasedCategory"])
    # print('jumlah Koin ', dataOutput["nominalUang"])
    # print("-"*30)
    # print('jumlah Koin ', dataOutput["jumlahKoinBasedCategory"]['koinSeratus'])
    # print('jumlah Koin ', dataOutput["jumlahKoinBasedCategory"]['koinDuaratus'])
    # print('jumlah Koin ', dataOutput["jumlahKoinBasedCategory"]['koinLimaratus'])
    # print('jumlah Koin ', dataOutput["jumlahKoinBasedCategory"]['koinSeribu'])
    cv2.waitKey(0) 
    # closing all open windows 
    cv2.destroyAllWindows() 

    return dataOutput