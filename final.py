import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('9020.jpg')
cap = cv2.VideoCapture('IMG_8276 5.mov')
NODISTORTION = True
VIDEO = True
IMAGE = False

framRate = 10 #bigger = slower

originalImg = img.copy()
height, width = originalImg.shape[:2]

groupOneContours = None
groupTwoContours = None
g1Vec = None
g2Vec = None

timeData = np.array([])
angleData = np.array([])
yPosData = np.array([])

ret, frame = cap.read()
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)


lowerOne = np.array([50, 150, 120])  # pink
upperOne = np.array([160, 200, 170])

lowerTwo = np.array([100, 60, 160])  # yellow
upperTwo = np.array([255, 160, 230])

#camera
camMatrix = np.array([[4.55771011e+03, 0.00000000e+00, 2.93944855e+03],
                      [0.00000000e+00, 4.53858466e+03, 2.23826534e+03],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Distortion coefficients
distCoeff = np.array([[ 0.02916309, -0.12321723, -0.00792241,  0.01961075,  0.07613373]])

newCamMatrix, roi = cv2.getOptimalNewCameraMatrix(camMatrix, distCoeff, (width, height), 1, (width, height))

if NODISTORTION:
    img = cv2.undistort(img, camMatrix, distCoeff, None, newCamMatrix)

img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def seeMasks():
    global groupOneContours, groupTwoContours, height, width

    groupOne = cv2.inRange(img, lowerOne, upperOne)
    groupOneContours, hierachy = cv2.findContours(groupOne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    groupOneContours = sorted(groupOneContours, key=cv2.contourArea, reverse=True)

    groupTwo = cv2.inRange(img, lowerTwo, upperTwo)
    groupTwoContours, hierarchy = cv2.findContours(groupTwo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    groupTwoContours = sorted(groupTwoContours, key=cv2.contourArea, reverse=True)

    maskGroupOne = cv2.inRange(img, lowerOne, upperOne)  # Masking the image to find our color
    maskGroupTwo = cv2.inRange(img, lowerTwo, upperTwo)  # Masking the image to find our color
    print(width, height)
    
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("yellow", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pink", cv2.WINDOW_NORMAL)

    cv2.imshow("original", originalImg)
    cv2.imshow("pink" , maskGroupOne)
    cv2.imshow("yellow", maskGroupTwo)
    cv2.waitKey(100)

    cv2.resizeWindow("original", 550, 550)
    cv2.resizeWindow("yellow", 550 , 550)
    cv2.resizeWindow("pink", 550, 550)

    cv2.moveWindow("original", 0, 100) 
    cv2.moveWindow("pink", 550, 100)
    cv2.moveWindow("yellow", 1000, 100)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def seeMasksVideo(frame):
    global groupOneContours, groupTwoContours
        
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    groupOne = cv2.inRange(lab, lowerOne, upperOne)
    groupOneContours, hierachy = cv2.findContours(groupOne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    groupOneContours = sorted(groupOneContours, key=cv2.contourArea, reverse=True)

    groupTwo = cv2.inRange(lab, lowerTwo, upperTwo)
    groupTwoContours, hierarchy = cv2.findContours(groupTwo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    groupTwoContours = sorted(groupTwoContours, key=cv2.contourArea, reverse=True)

    maskGroupOne = cv2.inRange(lab, lowerOne, upperOne)  # Masking the image to find our color

    maskGroupTwo = cv2.inRange(lab, lowerTwo, upperTwo)  # Masking the image to find our color

    maskFinal = maskGroupTwo | maskGroupOne
    return maskFinal

def groupOneCalc():
    global g1Vec, groupOneCentroids

    groupOneCentroids = []

    for contour in groupOneContours[:2]:
        # Calculate moments of the contour
        M = cv2.moments(contour)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Append centroid coordinates to the list
        groupOneCentroids.append((cX, cY))

        # Draw a circle at the centroid
        cv2.circle(img, (cX, cY), 5, (0, 255, 0), 10)


    g1Vec = np.array((-groupOneCentroids[1][0] + groupOneCentroids[0][0], -groupOneCentroids[1][1] + groupOneCentroids[0][1]))

    return g1Vec

def groupTwoCalc():
    #these r lower contours 
    global g2Vec, groupTwoCentroids

    groupTwoCentroids = []
    for contour in groupTwoContours[:2]:
        # Calculate moments of the contour
        M = cv2.moments(contour)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Append centroid coordinates to the list
        groupTwoCentroids.append((cX, cY))

        # Draw a circle at the centroid
        cv2.circle(img, (cX, cY), 5, (0, 255, 0), 10)

    g2Vec = np.array((-groupTwoCentroids[1][0] + groupTwoCentroids[0][0], -groupTwoCentroids[1][1] + groupTwoCentroids[0][1]))

    return g2Vec

def calcAngle():
    global img
    dotProduct = np.dot(g1Vec, g2Vec)
    magG1 = np.linalg.norm(g1Vec)
    magG2 = np.linalg.norm(g2Vec)
    angle = np.degrees(np.arccos((dotProduct) / (magG1 * magG2)))
    return angle

def main():
    global img, timeData, yPosData, angleData
    frameIndex = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps is: ",  fps)

    if IMAGE:    
        seeMasks()
        groupOneCalc()
        groupTwoCalc()
        angle = calcAngle()
        img = cv2.putText(img, "Angle is: " + str(angle), (200, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:

        while True:
            ret, frame = cap.read()
            time = frameIndex / fps
            if not ret:
                print("End of vid")
                break
            
            timeData = np.append(timeData, time)

            imgV = seeMasksVideo(frame)
            heightV, widthV = imgV.shape[:2]

            groupOneCalc()
            groupTwoCalc()
            angle = calcAngle()
            angleData = np.append(angleData, angle)
            yPosData = np.append(yPosData, groupTwoCentroids[1][1])
            imgV = cv2.putText(imgV, 'angle: ' + str(int(angle)), (200, heightV - 20), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5, cv2.LINE_AA)

            cv2.circle(frame, (groupTwoCentroids[0][0], groupTwoCentroids[0][1]), 5, (255, 255, 255), 10)
            cv2.circle(frame, (groupTwoCentroids[1][0], groupTwoCentroids[1][1]), 5, (255, 255, 255), 10)
            
            cv2.circle(frame, (groupOneCentroids[0][0], groupOneCentroids[0][1]), 5, (0, 255, 0), 10)
            cv2.circle(frame, (groupOneCentroids[1][0], groupOneCentroids[1][1]), 5, (0, 255, 0), 10)

           
            cv2.namedWindow("masked", cv2.WINDOW_NORMAL)
            cv2.namedWindow("original", cv2.WINDOW_NORMAL)

            cv2.imshow('masked', imgV)
            cv2.imshow("original", frame )

            cv2.resizeWindow("masked", 700, 400)
            cv2.resizeWindow("original", 700 , 400)

            cv2.moveWindow("masked", 0, 2)
            cv2.moveWindow("original", 800, 2)
            frameIndex += 1
            if cv2.waitKey(framRate) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        fig, axs = plt.subplots(2,1, figsize = (10,8))
        axs[1].plot(timeData, angleData, label="Angle v Time")
        axs[1].set_title("Angle v Time")
        axs[0].plot(timeData,yPosData, label="Yposition v Time")
        axs[0].set_title("Yposition v Time")
        plt.show()

if __name__ == "__main__":
    main()