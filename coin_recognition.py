import numpy as np
import glob
import cv2
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def coin_recog() :
    image = cv2.imread("uploads/input_image.jpg") # read in image

    # resize image while retaining aspect ratio
    print(image.shape[1])
    d = 1024 / image.shape[1]
    print(d)
    dim = (1024, int(image.shape[0] * d))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    output = image.copy() # create a copy of the image to display results
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert image to grayscale

    # create a CLAHE object to apply contrast limiting adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit = 6.0, tileGridSize = (8, 8))
    gray = clahe.apply(grayscale)

    def calc_histogram(img):
        m = np.zeros(img.shape[:2], dtype="uint8") # create mask
        (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
        cv2.circle(m, (w, h), 60, 255, -1) # draw circle
        # calcHist expects a list of images, color channels, mask, bins, ranges
        h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        return cv2.normalize(h, h).flatten() # return normalized "flattened" histogram

    def calc_hist_from_file(file):
        img = cv2.imread(file)
        return calc_histogram(img) # return histogram


    # define Enum class
    class Enum(tuple):
        __getattr__ = tuple.index # define __getattr__ to return index of tuple

    # Enumerate material types for use in classifier
    Material = Enum(('lima_ratus', 'seribu', 'seratus', 'dua_ratus'))

    # locate sample image files
    sample_images_lima_ratus = glob.glob("sample_images/lima_ratus/*.png")
    sample_images_seribu = glob.glob("sample_images/seribu/*.png")
    sample_images_seratus = glob.glob("sample_images/seratus/*.png")
    sample_images_dua_ratus = glob.glob("sample_images/dua_ratus/*.png")

    # define training data and labels
    X = []
    y = []

    # compute and store training data and labels
    for i in sample_images_lima_ratus:
        X.append(calc_hist_from_file(i))
        y.append(Material.lima_ratus)
    for i in sample_images_seribu:
        X.append(calc_hist_from_file(i))
        y.append(Material.seribu)
    for i in sample_images_seratus:
        X.append(calc_hist_from_file(i))
        y.append(Material.seratus)
    for i in sample_images_dua_ratus:
        X.append(calc_hist_from_file(i))
        y.append(Material.dua_ratus)

    # instantiate classifier
    classifier = MLPClassifier(solver = "adam", hidden_layer_sizes=(1000,), max_iter=100000,learning_rate="adaptive")

    # split samples into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

    # train and score classifier
    classifier.fit(X_train, y_train) # train classifier
    score = int(classifier.score(X_test, y_test) * 100) # calculate score
    print("Classifier mean accuracy: ", score) # print accuracy

    blurred = cv2.GaussianBlur(gray, (7, 7), 0) # blur image
    # apply edge detection to image
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp = 2.2, minDist = 100, param1 = 200, param2 = 100, minRadius = 50, maxRadius = 120)

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
        for (x, y, r) in circles[0, :]:
            diameter.append(r)

        circles = np.round(circles[0, :]).astype("int") # convert coordinates and radii to integers

        for (x, y, d) in circles: # loop over coordinates and radii of the circles
            count += 1

            coordinates.append((x, y)) # add coordinates to list
            roi = image[y - d:y + d, x - d:x + d] # extract region of interest
            material = predictMaterial(roi) # predict material type
            materials.append(material) # add material type to list
            # draw contour and results in the output image
            cv2.circle(output, (x, y), d, (0, 255, 0), 2)
            cv2.putText(output, material, (x - 40, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness = 2, lineType = cv2.LINE_AA)

    # get biggest diameter
    biggest = max(diameter)
    i = diameter.index(biggest) # get index of biggest diameter
    print(diameter) # print diameters

    # scale everything according to maximum diameter
    if materials[i] == "seribu":
        diameter = [x / biggest * 20.75 for x in diameter]
        scaledTo = "Skala berdsasarkan koin 1.000"
    elif materials[i] == "lima_ratus":
        diameter = [x / biggest * 15.25 for x in diameter]
        scaledTo = "Skala berdsasarkan koin 500"
    elif materials[i] == "seratus":
        diameter = [x / biggest * 18.25 for x in diameter]
        scaledTo = "Skala berdsasarkan koin 100"
    elif materials[i] == "dua_ratus":
        diameter = [x / biggest * 20.25 for x in diameter]
        scaledTo = "Skala berdsasarkan koin 200"
    else:
        scaledTo = "unable to scale.."

    i = 0
    total = 0
    countCoinBasedCategory = {
        'koinSeratus': 0,
        'koinDuaratus': 0,
        'koinLimaratus': 0,
        'koinSeribu': 0,
        }
    nominalUang = 0
    
    while i < len(diameter): # loop over diameters
        d = diameter[i] # get diameter
        print('d = ',d)
        print('len d = ',len(diameter))
        print('i = ',i)
        m = materials[i] # get material type
        (x, y) = coordinates[i] # get coordinates
        if m == "lima_ratus": # if material is lima_ratus peso
            t = "500"
            countCoinBasedCategory["koinLimaratus"] += 1
            nominalUang += 500
        elif m == "seribu": # if material is five cents
            t = "1.000"
            countCoinBasedCategory["koinSeribu"] += 1
            nominalUang += 1000
        elif m == "seratus": # if material is seratus peso
            t = "100"
            countCoinBasedCategory["koinSeratus"] += 1
            nominalUang += 100
        elif m == "dua_ratus": # if material is dua_ratus peso
            t = "200"
            countCoinBasedCategory["koinDuaratus"] += 1
            nominalUang += 200
        else: # if material is unknown
            t = "Unknown"

        # write result on output image
        cv2.putText(output, t, (x - 40, y + 22), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness = 2, lineType = cv2.LINE_AA)
        i += 1

    # resize output image while retaining aspect ratio
    d = 800 / output.shape[1] # calculate resize factor
    dim = (800, int(output.shape[0] * d)) # get dimensions
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA) # resize image
    output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA) # resize output image

    # write summary on output image
    cv2.putText(output, scaledTo, (5, output.shape[0] - 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), lineType = cv2.LINE_AA)
    cv2.putText(output, "Jumlah Koin : {}".format(count, total / 100), (5, output.shape[0] - 24), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), lineType = cv2.LINE_AA)
    cv2.putText(output, "Classifier mean accuracy: {}%".format(score), (5, output.shape[0] - 8), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), lineType = cv2.LINE_AA)
    
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
    cv2.destroyAllWindows()
    return dataOutput