import SimpleITK as sitk
import numpy as np
from collections import Counter

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

def EvaluateBreastSide(sitkVolume, pectoralSideVolume):

    sitkVolume = sitk.Normalize(sitkVolume)
    raisingL, raisingR, topPointL, topPointR, v1 = findRaisingHeightAndCreateV1(sitkVolume, pectoralSideVolume)

    # Initialize the datatype as UInt8
    resVolume = sitk.Cast(sitkVolume, sitk.sitkUInt8)

    # run slice with index=middle to index=0
    resVolume = SliceToMask(sitkVolume, resVolume, topPointL, topPointR, v1, pectoralSideVolume, forward=False)

    # run slice with index=middle to index=last
    resVolume = SliceToMask(sitkVolume, resVolume, topPointL, topPointR, v1, pectoralSideVolume, forward=True)

    return resVolume, v1, raisingL, raisingR

def SliceToMask(sitkVolume, resVolume, topPointL, topPointR, v1, pectoralSideVolume, forward=False):
    """
    In this function, it call the findNipple function to retain nipple region in CannyEdgeDetection, and call preprocess function (it maybe loss nipple region).
    After that, using sitk.Or combine the two mask as mentioned above.
    Finally, call the findV1 funciton.
    """
    
    # Get volume size
    volumeSize = list(sitkVolume.GetSize())

    if(forward == True):
        start = volumeSize[2]//2 +1
        end = volumeSize[2]
        direction = 1
    else:
        start = volumeSize[2]//2
        end = -1
        direction = -1

    for sliceNum in range(start, end, direction):
        imageSlice = sitkVolume[0:volumeSize[0], 0:volumeSize[1], sliceNum]
        pectoralSideMask = pectoralSideVolume[0:volumeSize[0], 0:volumeSize[1], sliceNum]
        
        res = preprocess(imageSlice, pectoralSideMask=pectoralSideMask, sliceLocation=(float(sliceNum)/float(volumeSize[2])), cutHeight=v1[volumeSize[2]//2].y, pixelSpacing_y=sitkVolume.GetSpacing()[1])
        resArray = sitk.GetArrayFromImage(res)
        if(sliceNum != volumeSize[2]//2):
            findV1(sliceNum, resArray, v1, topPointL = topPointL, topPointR = topPointR, direction = direction)
        
        res = sitk.GetImageFromArray(resArray)
        resVolume = sitk.Paste(
            destinationImage = resVolume,
            sourceImage = sitk.JoinSeries(res),
            sourceSize = [volumeSize[0], volumeSize[1], 1],
            sourceIndex = [0, 0, 0],
            destinationIndex = [0, 0, sliceNum])
    return resVolume

def preprocess(image, pectoralSideMask=None, sliceLocation=0.4, cutHeight=0, pixelSpacing_y=0.625):
    edge = edgeDetection(image, nipple = False)
    imageSize = list(image.GetSize())

    if(cutHeight!=0):
        edge_nipple = edgeDetection(image, cutHeight=cutHeight, sliceLocation=sliceLocation, nipple = True)
        edge = sitk.Or(edge, edge_nipple)
    else:
        cutHeight = int(imageSize[0]//2)

    if(pectoralSideMask!=None):
        edge = sitk.Or(edge, pectoralSideMask)

    image = fillHole(edge, cutHeight, pixelSpacing_y)

    return image

def edgeDetection(image, cutHeight=0, sliceLocation=0.4, nipple = False):
    if(nipple == True):
        lowerThreshold = 0.3 + 0.1*abs(sliceLocation-0.4)*1.7
        upperThreshold = 0.12 +0.08*abs(sliceLocation-0.4)*1.7
    else:
        lowerThreshold = 0.4
        upperThreshold = 0.2
    image = sitk.CurvatureAnisotropicDiffusion(image, 
                                            timeStep = 0.0625,
                                            conductanceParameter = 3,
                                            conductanceScalingUpdateInterval = 1,
                                            numberOfIterations = 3)

    image = sitk.CannyEdgeDetection(image, 
                                    lowerThreshold = lowerThreshold, 
                                    upperThreshold = upperThreshold,                                  # avoid noise
                                    variance = (1, 1),
                                    )

    image = sitk.Cast(image, sitk.sitkUInt8)
    
    if(nipple == True):
        imageSize = list(image.GetSize())
        interesting = sitk.RegionOfInterest(image,                                              # catch nipple region
                                    size = (int(imageSize[0]), cutHeight),
                                    index = (0, 0))

        image = sitk.ConstantPad(image1 = interesting,                                          # zero-padding
                                    padLowerBound = (0, 0),
                                    padUpperBound = (0, int(imageSize[1]) - cutHeight),
                                    constant = 0)

    return image

def fillHole(image, cutHeight, pixelSpacing_y=0.625):
    radius = int(12.5/pixelSpacing_y)
    image = sitk.BinaryDilate(image, (radius, radius), sitk.sitkBall, 0, 1, False)                  # Dilation
    image = padOne(image, cutHeight)
    image = sitk.BinaryFillhole(image, False, 1)
    image = sitk.BinaryErode(image, (radius, radius), sitk.sitkBall, 0, 1, False)                   # Erosion

    return image

def padOne(image, cutHeight):
    imageSize = list(image.GetSize())
    array = sitk.GetArrayFromImage(image)

    array[cutHeight+(int(imageSize[1]-1)-cutHeight)//2, :] = 1
    array[cutHeight:int(imageSize[1]-1), 0] = 1
    array[cutHeight:int(imageSize[1]-1), int(imageSize[0]-1)] = 1
    image = sitk.GetImageFromArray(array)

    return image

def findV1(sliceNum, image, v1, topPointL=Point(0,0), topPointR=Point(0,0), direction = 0):
    """
    find v1 in image[start_x : end_x][start_y : end_y]
    Notice: limits start_x and end_x in topPointL to topPointR
    """
    if(direction==0):                                                                       # middle slice in z-axis
        start_x = int(image.shape[0])//3
        end_x = (int(image.shape[0])//3)*2
        topPointR.x = int(image.shape[0])-1
    else:                                                                                    # others, direction = 1 or -1
        start_x = v1[sliceNum - direction].x
        end_x = v1[sliceNum - direction].x+2

    start_y = max(topPointL.y, topPointR.y)
    end_y = int(image.shape[1])-1
    leftBoundary_x = topPointL.x
    rightBoundary_x = topPointR.x

    if(start_x < leftBoundary_x or end_x > rightBoundary_x):
        start_x = leftBoundary_x
        end_x = rightBoundary_x

    v1_x = 0
    v1_y = 0
    for i in range(start_x, end_x):
        for j in range(start_y, end_y):
            if(int(image[j][i]) > 0):
                if(j >= v1_y):
                    v1_x = i
                    v1_y = j
                break
                
    v1[sliceNum].x = v1_x
    v1[sliceNum].y = v1_y
    return

def findSingleV1(image):
    start_x = int(image.shape[0])//3
    end_x = (int(image.shape[0])//3)*2
    end_y = int(image.shape[1])-1

    v1_x = 0
    v1_y = 0
    for i in range(start_x, end_x):
        for j in range(0, end_y):
            if(int(image[j][i]) > 0):
                if(j >= v1_y):
                    v1_x = i
                    v1_y = j
                break
                
    return v1_x, v1_y

def islandRemoval2D(image):
    rows = list(image.GetSize())[0]
    cols = list(image.GetSize())[1]
    label = sitk.ConnectedComponent(image)
    labelArray = sitk.GetArrayFromImage(label)

    count_array = []
    for col in range(cols):
        count_array += list(labelArray[col])
    count = Counter(count_array)                        # Counter : get [(value1, count1), (value2, count2), (value3, count3) ...]

    if(len(count.most_common(2)) == 2):
        if(count.most_common(2)[0][0] == 0):            # count.most_common(3) : get top3 (value,count) pairs that maybe including (0, count)
            one = count.most_common(2)[1][0]
        elif(count.most_common(2)[1][0] == 0):
            one = count.most_common(2)[0][0]

        for y in range(rows):
            for x in range(cols):
                com = labelArray[x][y]
                if(com != one):
                    labelArray[x][y] = 0
                else:
                    labelArray[x][y] = 1
    # Only 0 or 1
    image = sitk.GetImageFromArray(labelArray)          
    image = sitk.Cast(image, sitk.sitkUInt8)

    return image

def findHighest2Point(image):
    get = False
    L_x=0
    L_y=0
    for i in range(int(image.shape[1])):
        for j in range(0, int(image.shape[0])//2):
            if(int(image[i][j]) > 0):
                L_x = j
                L_y = i
                get = True
        if(get):
            break

    get = False
    R_x=int(image.shape[1]-1)
    R_y=0
    for i in range(int(image.shape[1])):
        for j in range(int(image.shape[0])//2, int(image.shape[0])):
            if(int(image[i][j]) > 0):
                R_x = j
                R_y = i
                get = True
        if(get):
            break
    return Point(L_x, L_y), Point(R_x, R_y)
        
def calculateRaising(topPointL, topPointR, v1, numOfSlice, pixelSpacing_y=0.625):
    #Then calculate raising height when get the topPoint and v1.
    diffSize = 0
    if(topPointL.y > 0 and topPointR.y > 0):
        diffSize = int(float(v1[numOfSlice].y - topPointL.y) / float(v1[numOfSlice].y - topPointR.y))
    raisingL = 1 + int( float(v1[numOfSlice].y - topPointL.y) / 20.0 / pixelSpacing_y) + diffSize 
    if(raisingL > 8.0 / pixelSpacing_y):
        raisingL = int(8.0 / pixelSpacing_y)

    if(topPointL.y > 0 and topPointR.y > 0):
        diffSize = int(float(v1[numOfSlice].y - topPointR.y) / float(v1[numOfSlice].y - topPointL.y))
    raisingR = 1 + int( float(v1[numOfSlice].y - topPointR.y) / 20.0 / pixelSpacing_y) + diffSize
    if(raisingR > 8.0 / pixelSpacing_y):
        raisingR = int(8.0 / pixelSpacing_y)
    
    return raisingL, raisingR

def findRaisingHeightAndCreateV1(sitkVolume, pectoralSideVolume):
    """
    Call islandRemoval2D to ensure that findHighest2Point function will not find any noise.
    After that, call calculateRaising function.
    """
    # Get volume size
    volumeSize = list(sitkVolume.GetSize())
    # Create a v1 object array
    v1 = [Point(volumeSize[0] // 2, 0) for i in range(volumeSize[2])]

    numOfSlice = volumeSize[2]//2
    imageSlice = sitkVolume[0:volumeSize[0], 0:volumeSize[1], numOfSlice]
    pectoralSideMask = pectoralSideVolume[0:volumeSize[0], 0:volumeSize[1], numOfSlice]

    res = preprocess(imageSlice, pectoralSideMask=pectoralSideMask, pixelSpacing_y=sitkVolume.GetSpacing()[1])
    res = islandRemoval2D(res)
    resArray = sitk.GetArrayFromImage(res)

    findV1(numOfSlice, resArray, v1)
    topPointL, topPointR = findHighest2Point(resArray)

    raisingL, raisingR = calculateRaising(topPointL, topPointR, v1, numOfSlice, pixelSpacing_y=sitkVolume.GetSpacing()[1])

    return raisingL, raisingR, topPointL, topPointR, v1

