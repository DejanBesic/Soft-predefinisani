import numpy as np
import cv2
from time import sleep
from keras import models
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import load_model
import matplotlib.pyplot as plt
import collections
import pickle
import roiForTheWin as rokiFTW
from scipy.spatial import distance


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 50, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

#ne radi
def select_roi_radi_100_posto(image_orig, image_bin, roiList, ann):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if h > 14 and h < 30 and w < 30:
            region = image_bin[y:y+h+1, x:x+w+1]
            resized = resize_region(region)
            test_inputs = []
            test_inputs.append(matrix_to_vector(scale_to_range(resized)))
            result = ann.predict(np.array(test_inputs, np.float32))
            alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            value = display_result(result, alphabet)
            if not roiList:
                roiList, ro = create_roi(roiList, x, y, w, h, value[0])
            else:
                roiList, ro = get_the_closest(roiList, x, y, w, h, value[0])
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][1])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions, roiList, ann

def create_roi(roiList, x, y, w, h, value):
    ro = rokiFTW.RoiFTW(value)
    ro.addDDXyCoords(x+w, y+h)
    ro.addDLxyCoords(x, y+h)
    ro.addGDXyCoords(x+w, y)
    ro.addGLxyCoords(x, y)
    roiList.append(ro)
    return roiList, ro

def get_the_closest(roiList, x, y, w, h, value):
    distances = []
    for i in range(len(roiList)):
        if roiList[i].value == value:
            dd = roiList[i].ddXyCoords
            length = len(dd)-1
            #dist = ((gl[length][0]-x)**2 + (gl[length][1]-y)**2)**1/2
            dist = distance.euclidean([x+w, y+h], dd[length])
            if dist < 30:
                distances.append([dist, i])
    distances = sorted(distances, key=lambda item: item[1])
    if not distances:
        return create_roi(roiList, x, y, w, h, value)
    else:
        ro = roiList[distances[0][1]]
        ro.addDDXyCoords(x+w, y+h)
        ro.addDLxyCoords(x, y+h)
        ro.addGDXyCoords(x+w, y)
        ro.addGLxyCoords(x, y)
        return roiList, ro

def select_roi_test(image_orig, image_bin, roiList, ann):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if h > 14 and h < 30 and w < 30:
            region = image_bin[y:y+h+1, x:x+w+1]
            resized = resize_region(region)
            test_inputs = []
            test_inputs.append(matrix_to_vector(scale_to_range(resized)))
            result = ann.predict(np.array(test_inputs, np.float32))
            alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            value = display_result(result, alphabet)
            key = h*10 + w*10 + value[0]*10000 + region.nbytes*100000
            dozvola = True
            if not roiList:
                r = roki.Roi(value, key)
                r.addXyCoords(x, y)
                roiList.append(r)
            else:
                for i in range(len(roiList)):
                    roiKey = roiList[i].getKey()
                    xyCoords = roiList[i].getXyCoords()
                    length = len(xyCoords)-1
                    if value[0] == roiList[i].getValue()[0]:
                        if x+30 >= xyCoords[length][0] and x-30 <= xyCoords[length][0] and \
                                y+30 >= xyCoords[length][1] and y-30 <= xyCoords[length][1]:
                            if roiKey == key:
                                roiList[i].addXyCoords(x, y)
                                dozvola = False
                                break
                if dozvola:
                    r = roki.Roi(value, key)
                    r.addXyCoords(x, y)
                    roiList.append(r)
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][1])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions, roiList

def is_in_XY_range(line, x, y, w, h):
    lineRange = False

    if (line[0] <= x and line[2] >= x and line[1] >= y and line[3] <= y) or \
         (line[0] <= x+w and line[2] >= x+w and line[1] >= y and line[3] <= y) or \
          (line[0] <= x and line[2] >= x and line[1] >= y+h and line[3] <= y+h) or \
           (line[0] <= x+w and line[2] >= x+w and line[1] >= y+h and line[3] <= y+h):
        return True
    else:
        return False

def is_near_line(line, x, y):
    Y = getY(line, x)
    difference = Y - y
    if difference >= 0 and difference <= 5:
        return True
    else:
        return False

def select_roi_win_combo(image_orig, image_bin, blueLine, greenLine, roiList, ann, suma):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if  h > 14 and h < 30 and w < 30:
            region = image_bin[y:y+h+1, x:x+w+1]
            resized = resize_region(region)
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
            test_inputs = []
            test_inputs.append(matrix_to_vector(scale_to_range(resized)))
            result = ann.predict(np.array(test_inputs, np.float32))
            alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            value = display_result(result, alphabet)
            dozvola = True
            key = h*10 + w*10 + value[0]*10000 + region.nbytes*100000
            if not roiList:    
                ro = rokiFTW.RoiFTW(value[0])
                ro.addDDXyCoords(x+w, y+h)
                ro.addDLxyCoords(x, y+h)
                ro.addGDXyCoords(x+w, y)
                ro.addGLxyCoords(x, y)
                ro.setKey(key)
                roiList.append(ro)
            else:
                for i in range(len(roiList)):
                    ro = roiList[i]
                    length = len(ro.ddXyCoords)-1
                    if value[0] == ro.value:
                        if  x + w + 5 >= ro.ddXyCoords[length][0] and \
                            x + w - 5 <= ro.ddXyCoords[length][0] and \
                            y + h + 5 >= ro.ddXyCoords[length][1] and \
                            y + h - 5 <= ro.ddXyCoords[length][1] and \
                            x + 5 >= ro.dlXyCoords[length][0] and \
                            x - 5 <= ro.dlXyCoords[length][0] and \
                            y + h + 5 >= ro.dlXyCoords[length][1] and \
                            y + h - 5 <= ro.dlXyCoords[length][1] and \
                            x + w + 5 >= ro.gdXyCoords[length][0] and \
                            x + w - 5 <= ro.gdXyCoords[length][0] and \
                            y + 5 >= ro.gdXyCoords[length][1] and \
                            y - 5 <= ro.gdXyCoords[length][1] and \
                            x + 5 >= ro.glXyCoords[length][0] and \
                            x - 5 <= ro.glXyCoords[length][0] and \
                            y + 5 >= ro.glXyCoords[length][1] and \
                            y - 5 <= ro.glXyCoords[length][1]:
                            ro.addDDXyCoords(x+w, y+h)
                            ro.addDLxyCoords(x, y+h)
                            ro.addGDXyCoords(x+w, y)
                            ro.addGLxyCoords(x, y)
                            #roiList[i], suma = calculate_sum(roiList[i], blueLine, greenLine, suma)
                            dozvola = False
                            break
                if dozvola:
                    dozvola1 = True
                    for i in range(len(roiList)):
                        ro = roiList[i]
                        length = len(ro.ddXyCoords)-1
                        if value[0] == ro.value:
                            if x + w + 30 >= ro.ddXyCoords[length][0] and \
                                x + w - 30 <= ro.ddXyCoords[length][0] and \
                                y + h + 30 >= ro.ddXyCoords[length][1] and \
                                y + h - 30 <= ro.ddXyCoords[length][1] and \
                                x + 30 >= ro.dlXyCoords[length][0] and \
                                x - 30 <= ro.dlXyCoords[length][0] and \
                                y + h + 30 >= ro.dlXyCoords[length][1] and \
                                y + h - 30 <= ro.dlXyCoords[length][1] and \
                                x + w + 30 >= ro.gdXyCoords[length][0] and \
                                x + w - 30 <= ro.gdXyCoords[length][0] and \
                                y + 30 >= ro.gdXyCoords[length][1] and \
                                y - 30 <= ro.gdXyCoords[length][1] and \
                                x + 30 >= ro.glXyCoords[length][0] and \
                                x - 30 <= ro.glXyCoords[length][0] and \
                                y + 30 >= ro.glXyCoords[length][1] and \
                                y - 30 <= ro.glXyCoords[length][1]:
                                ro.addDDXyCoords(x+w, y+h)
                                ro.addDLxyCoords(x, y+h)
                                ro.addGDXyCoords(x+w, y)
                                ro.addGLxyCoords(x, y)
                                #roiList[i], suma = calculate_sum(roiList[i], blueLine, greenLine, suma)
                                dozvola1 = False
                                break
                    if dozvola1:
                        ro = rokiFTW.RoiFTW(value[0])
                        ro.addDDXyCoords(x+w, y+h)
                        ro.addDLxyCoords(x, y+h)
                        ro.addGDXyCoords(x+w, y)
                        ro.addGLxyCoords(x, y)
                        ro.setKey(key)
                        roiList.append(ro)
    regions_array = sorted(regions_array, key=lambda item: item[1][1])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions, roiList, suma

def select_roi_numbers(image_orig, image_bin, blueLine, greenLine,  roiList, ann, suma):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if  h > 14 and h < 30 and w < 30:
            region = image_bin[y:y+h+1, x:x+w+1]
            resized = resize_region(region)
            test_inputs = []
            test_inputs.append(matrix_to_vector(scale_to_range(resized)))
            result = ann.predict(np.array(test_inputs, np.float32))
            alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            value = display_result(result, alphabet)
            key = h*10 + w*10 + value[0]*10000 + region.nbytes*100000
            dozvola = True
            if not roiList:
                r = roki.Roi(value, key)
                r.addXyCoords(x, y)
                roiList.append(r)
            else:
                for i in range(len(roiList)):
                    roiKey = roiList[i].getKey()
                    xyCoords = roiList[i].getXyCoords()
                    length = len(xyCoords)-1
                    if value[0] == roiList[i].getValue()[0]:
                        if x+30 >= xyCoords[length][0] and x-30 <= xyCoords[length][0] and \
                                y+30 >= xyCoords[length][1] and y-30 <= xyCoords[length][1]:
                            if roiKey == key:
                                if is_near_line(blueLine, x, y, w, h) and roiList[i].getIntersectsBlue() == False:
                                    roiList[i].setIntersectsBlue(True)
                                    suma = suma + value[0]
                                if is_near_line(greenLine, x, y, w, h) and roiList[i].getIntersectsGreen() == False:
                                    roiList[i].setIntersectsGreen(True)
                                    suma = suma - value[0]
                                roiList[i].addXyCoords(x, y)
                                dozvola = False
                                break
                if dozvola:
                    if is_near_line(blueLine, x, y, w, h) or is_near_line(greenLine, x, y, w, h):
                        r = roki.Roi(value, key)
                        r.addXyCoords(x, y)
                        roiList.append(r)
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x,y,w,h)])
            cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][1])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions, roiList, suma

def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    lista_brojeva = []
    for contour in contours: 
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if h<20 and w<20 and h>10 and w>2:
            region = image_bin[y:y+h+1,x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if y<100:
                lista_brojeva.append(0)
            if y<200 and y >=100:
                lista_brojeva.append(1)
            if y<300 and y >=200:
                lista_brojeva.append(2)
            if y<400 and y >=300:
                lista_brojeva.append(3)
            if y<500 and y >=400:
                lista_brojeva.append(4)
            if y<600 and y >=500:
                lista_brojeva.append(5)
            if y<700 and y >=600:
                lista_brojeva.append(6)
            if y<800 and y >=700:
                lista_brojeva.append(7)
            if y<900 and y >=800:
                lista_brojeva.append(8)
            if y>=900:
                lista_brojeva.append(9)

    lista_brojeva = sorted(lista_brojeva)
    regions_array = sorted(regions_array, key=lambda item: item[1][1])
    sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions, lista_brojeva

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(10)
        i = alphabet[index]%10
        output[i] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32) # dati ulazi
    y_train = np.array(y_train, np.float32) # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose=0, shuffle=False)
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def recognize_number(number_img):
    result = ann.predict(number_img)
    return display_result(result, alphabet)

def create_train_save_ann():
    image_color = load_image('images/TNN.png')
    #image_color = cv2.cvtColor(poslednjiFrame, cv2.COLOR_BGR2RGB)
    img = invert(image_bin(image_gray(image_color)))
    #img_bin = erode(dilate(img)) nema suma pa nije potrebno
    #selected_regions je slika sa oznacenim roi, numbers je svaki roi posebno, lista_brojeva je lista slika numbers poredjanom po istom redosledu
    selected_regions, numbers, lista_brojeva = select_roi(image_color.copy(), img) 
    #moraju biti iste duzine slike brojeva i njihove vrednosti
    print(len(numbers))
    print(len(lista_brojeva))
    
    cv2.imshow('frame', selected_regions)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    inputs = prepare_for_ann(numbers)
    outputs = convert_output(lista_brojeva)
    #print outputs[1500:1550]
    ann = create_ann()
    ann = train_ann(ann, inputs, outputs)
    #cuvanje neuronske mreze kao neuralNetwork u okviru projekta
    with open('neuralNetwork2', 'wb') as output: pickle.dump(ann, output, pickle.HIGHEST_PROTOCOL)

def load_trained_ann():
    #ucitavanje neuronske mreze
    with open('neuralNetwork3', 'rb') as input: ann = pickle.load(input)
    #with open('neuralNetwork2', 'rb') as input: ann = pickle.load(input)
    return ann

def get_line_coords(filename):
    cap = cv2.VideoCapture(filename)
    #liste svih koordinata plave i zelene linije dobijenih tokom prolaska kroz video
    lista1x = []
    lista1y = []
    lista3x = []
    lista3y = []
    #poslednjiFrame = cap.read()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            #izdvajanje posebno plave linije pravljenjem maske
            lower_blue = np.array([130, 0, 0])
            upper_blue = np.array([255, 50, 50])
            mask = cv2.inRange(frame, lower_blue, upper_blue)
            frame1 = cv2.bitwise_and(frame, frame, mask=mask)
        
            #izdvajanje posebno zelene linije pravljenjem maske
            lower_green = np.array([0, 130, 0])
            upper_green = np.array([50, 255, 50])
            mask = cv2.inRange(frame, lower_green, upper_green)
            frame2 = cv2.bitwise_and(frame, frame, mask=mask)
            #sum je takodje zelene boje pa uklanjam erozijom i dilatacijom
            frame2 = dilate(erode(frame2))

            #pronalazim linije prvo plavih pa zelenih komponenti na slici
            edges1 = cv2.Canny(frame1, 0, 200, None, 3, True)
            edges2 = cv2.Canny(frame2, 0, 200, None, 3, True)
            minLineLength = 50
            maxLineGap = 0
            lines = cv2.HoughLinesP(edges1,1,np.pi/180,100,minLineLength,maxLineGap)
            for x1,y1,x2,y2 in lines[0]:
                #cuvam koordinate dobijene plave linije u liste
                lista1x.append(x1)
                lista1y.append(y1)
            lines = cv2.HoughLinesP(edges2,1,np.pi/180,100,minLineLength,maxLineGap)
            for x1,y1,x2,y2 in lines[0]:
                #cuvam koordinate dobijene zelene linije u liste
                lista3x.append(x1)
                lista3y.append(y1)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            #poslednjiFrame = frame
        else:
            break
    #pronalazim pocetnu i kranju koordinatu plave i zelene linije
    blueLine = [min(lista1x), max(lista1y), max(lista1x), min(lista1y)]
    greenLine = [min(lista3x), max(lista3y), max(lista3x), min(lista3y)]
    #cv2.line(poslednjiFrame, (min(lista1x), max(lista1y)), (max(lista1x), min(lista1y)), (0, 0, 255), 2)
    #cv2.line(poslednjiFrame, (min(lista3x), max(lista3y)), (max(lista3x), min(lista3y)), (0, 0, 255), 2)
    #cv2.imshow('frame', poslednjiFrame)
    #cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    return blueLine, greenLine

#Y = ((y2 - y1)/(x2 - x1))*(X - x1) + y1
def getY(line, x):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    X = x
    y2y1 = y2-y1
    x2x1 = x2-x1
    xx1 = X-x1
    y2y1xx1 = y2y1 * xx1
    y2y1xx1x2x1 = y2y1xx1 / x2x1
    Y = y2y1xx1x2x1 + y1
    return Y

def is_above_line(line, x, y):
    Y = getY(line, x)
    if Y >= y:
        return True
    else:
        return False

def calculate_sum(roiList, blueLine, greenLine, suma):

    gl = roiList.glXyCoords
    gd = roiList.gdXyCoords
    dl = roiList.dlXyCoords
    dd = roiList.ddXyCoords
    length = len(gl) - 1
 
    if is_near_line(blueLine, dl[length][0], dl[length][1]) and roiList.intersectBlue== False:
        roiList.setIntersectsBlue(True)
        #print '+' + str(roiList.value)
        suma = suma + roiList.value
    if is_near_line(blueLine, dd[length][0], dd[length][1]) and roiList.intersectBlue == False:
        roiList.setIntersectsBlue(True)
        #print '+' + str(roiList.value)
        suma = suma + roiList.value
    
    if is_near_line(greenLine, dl[length][0], dl[length][1]) and roiList.intersectGreen == False:
        roiList.setIntersectsGreen(True)
        #print '-' + str(roiList.value)
        suma = suma - roiList.value
    if is_near_line(greenLine, dd[length][0], dd[length][1]) and roiList.intersectGreen == False:
        roiList.setIntersectsGreen(True)
        #print '-' + str(roiList.value)
        suma = suma - roiList.value
    return roiList, suma

def calc_za_22(roiList, blueLine, greenLine ):
    for i in range(len(roiList)):
        gl = roiList[i].glXyCoords
        gd = roiList[i].gdXyCoords
        dl = roiList[i].dlXyCoords
        dd = roiList[i].ddXyCoords
        upperHalfBlue = False
        lowerHalfBlue = False
        upperHalfGreen = False
        lowerHalfGreen = False
        for k in range(len(gl)):
            w = gd[k][0] - gl[k][0]
            h = gd[k][1] - dd[k][1]
            if roiList[i].intersectBlue == False and is_in_XY_range(blueLine, gl[k][0], gl[k][1], w, h):
                if is_above_line(blueLine, gl[k][0], gl[k][1]) or is_above_line(blueLine, gd[k][0], gd[k][1]) or\
                    is_above_line(blueLine, dl[k][0], dl[k][1]) or is_above_line(blueLine, dd[k][0], dd[k][1]):
                    upperHalfBlue = True
                else:
                    lowerHalfBlue = True

                if lowerHalfBlue == True and upperHalfBlue == True:
                    roiList[i].setIntersectsBlue(True)
                    suma = suma + roiList[i].value
                    #print '+ ' + str(roiList[i].value)
            if roiList[i].intersectGreen == False and is_in_XY_range(greenLine, gl[k][0], gl[k][1], w, h):
                if is_above_line(greenLine, gl[k][0], gl[k][1]) or is_above_line(greenLine, gd[k][0], gd[k][1]) or\
                    is_above_line(greenLine, dl[k][0], dl[k][1]) or is_above_line(greenLine, dd[k][0], dd[k][1]):
                    upperHalfGreen = True
                else:
                    lowerHalfGreen = True
                if lowerHalfGreen == True and upperHalfGreen == True:
                    roiList[i].setIntersectsGreen(True)
                    suma = suma - roiList[i].value
                    #print '- ' + str(roiList[i].value)
    return suma, roiList

def final_score(roiList, blueLine, greenLine):
    suma = 0
    for i in range(len(roiList)):
        gl = roiList[i].glXyCoords
        dd = roiList[i].ddXyCoords
        
        for k in range(len(gl)-1):
            dist = distance.euclidean(gl[k], gl[k+1])
            w = dd[k][0]-gl[k][0]
            h = dd[k][1]-gl[k][1]
            if is_above_line(blueLine, gl[k][0], gl[k][1]) and is_above_line(blueLine, gl[k+1][0], gl[k+1][1]) == False and\
                roiList[i].intersectBlue == False and is_in_XY_range(blueLine, gl[k][0], gl[k][1], w, h) and is_in_XY_range(blueLine, gl[k+1][0], gl[k+1][1], w, h):
                roiList[i].setIntersectsBlue(True)
                suma = suma + roiList[i].value
                #print '+ ' + str(roiList[i].value)
            
            if is_above_line(greenLine, gl[k][0], gl[k][1]) and is_above_line(greenLine, gl[k+1][0], gl[k+1][1]) == False and\
                roiList[i].intersectGreen == False and is_in_XY_range(greenLine, gl[k][0], gl[k][1], w, h) and is_in_XY_range(greenLine, gl[k+1][0], gl[k+1][1], w, h):
                roiList[i].setIntersectsGreen(True)
                suma = suma - roiList[i].value
                #print '- ' + str(roiList[i].value)
    return suma, roiList

def main():
    ann = load_trained_ann()
    file = open('videos/out.txt','w') 
    file.write('RA 225/2014 Dejan Besic') 
    file.write('file    sum') 
    for f in range(10):
        filename = 'videos/video-' + str(f) + '.avi'
        blueLine, greenLine = get_line_coords(filename)
        cap = cv2.VideoCapture(filename)
        roiList = []
        suma = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                image_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = invert(image_bin(image_gray(image_color)))
                #57% tacnosti
                selected_regions, numbers, roiList, suma = select_roi_win_combo(image_color.copy(), img, blueLine, greenLine, roiList, ann, suma)
            else: 
                break
        cap.release()
        cv2.destroyAllWindows()
        suma, roiList = final_score(roiList, blueLine, greenLine)
        string = 'video-' + str(f) + '.avi ' + str(suma)
        print string
        file.write(string) 
    file.close()

if __name__ == '__main__':
    main()
    #print len(roiList)
    # listica = []
    # for lk in range(len(roiList)):
    #     if len(roiList[lk].ddXyCoords) > 5:
    #         listica.append(roiList[lk])
           # print "Ide u listu " + str(roiList[lk].value)
        #else: 
           # print "Ne ide u listu " + str(roiList[lk].value)


    # print len(listica)
    # roiList = listica
    

    # for i in range(len(roiList)):
    #     gl = roiList[i].glXyCoords
    #     gd = roiList[i].gdXyCoords
    #     dl = roiList[i].dlXyCoords
    #     dd = roiList[i].ddXyCoords
    #     upperHalfBlue = False
    #     lowerHalfBlue = False
    #     upperHalfGreen = False
    #     lowerHalfGreen = False
    #     for k in range(len(gl)):
    #         w = gd[k][0] - gl[k][0]
    #         h = gd[k][1] - dd[k][1]
    #         if roiList[i].intersectBlue == False and is_in_XY_range(blueLine, gl[k][0], gl[k][1], w, h):
    #             if is_above_line(blueLine, gl[k][0], gl[k][1]) or is_above_line(blueLine, gd[k][0], gd[k][1]) or\
    #                 is_above_line(blueLine, dl[k][0], dl[k][1]) or is_above_line(blueLine, dd[k][0], dd[k][1]):
    #                 upperHalfBlue = True
    #             else:
    #                 lowerHalfBlue = True

    #             if lowerHalfBlue == True and upperHalfBlue == True:
    #                 roiList[i].setIntersectsBlue(True)
    #                 suma = suma + roiList[i].value
    #                 #print '+ ' + str(roiList[i].value)
    #         if roiList[i].intersectGreen == False and is_in_XY_range(greenLine, gl[k][0], gl[k][1], w, h):
    #             if is_above_line(greenLine, gl[k][0], gl[k][1]) or is_above_line(greenLine, gd[k][0], gd[k][1]) or\
    #                 is_above_line(greenLine, dl[k][0], dl[k][1]) or is_above_line(greenLine, dd[k][0], dd[k][1]):
    #                 upperHalfGreen = True
    #             else:
    #                 lowerHalfGreen = True
    #             if lowerHalfGreen == True and upperHalfGreen == True:
    #                 roiList[i].setIntersectsGreen(True)
    #                 suma = suma - roiList[i].value
    #                 #print '- ' + str(roiList[i].value)


    
   
    #cap.release()
    #cv2.destroyAllWindows()
#file.close() 


    #print 'Suma za ' + filename + ' je: ' + str(suma)
