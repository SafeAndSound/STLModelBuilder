import SimpleITK as sitk
import numpy as np
from MRIBreastVolumeFunctions.BreastSideModule import Point
from collections import Counter

def EvaluateSideBoundary(inputImage, v1Array, raisingL, raisingR):
    volumeSize = list(inputImage.GetSize())                                                                 # Get volume size
    outputImage = sitk.Image(inputImage)
    sitkVolume = sitk.Cast(inputImage, sitk.sitkUInt8)

    highestPoint = Point(0, 0)                                                                       # Highest Point
    cutPointL = [Point(0, 0) for i in range(volumeSize[2])]                                          # Create a cutPointL array
    cutPointR = [Point(volumeSize[0]-1, volumeSize[1]-1) for i in range(volumeSize[2])]              # Create a cutPointR array

    ''' Operate in Axial side '''
    sliceCount = volumeSize[2]
    for sliceNum in range(sliceCount):
        imgArray = sitk.GetArrayFromImage(inputImage[0:volumeSize[0], 0:volumeSize[1], sliceNum])

        if sliceNum == volumeSize[2] // 2:
            highestPoint.y = findOneHighestPoint(imgArray)

        findCutPoint(sliceNum, imgArray, v1Array[sliceNum], cutPointL, cutPointR, raisingL, raisingR)

        resArray = splitPectoralFromBreast(sliceNum, imgArray, cutPointL, cutPointR)
        res = sitk.GetImageFromArray(resArray)

        sitkVolume = sitk.Paste(
            destinationImage = sitkVolume,
            sourceImage = sitk.JoinSeries(res),
            sourceSize = [volumeSize[0], volumeSize[1], 1],
            sourceIndex = [0, 0, 0],
            destinationIndex = [0, 0, sliceNum])

    ''' Operate in Axial side '''

    ''' Operate in Coronal side '''

    halfSlice = volumeSize[2] // 2
    sliceSelected = v1Array[halfSlice].y - ((v1Array[halfSlice].y - highestPoint.y) // 5)
    circleShapeImageSliceSelected = v1Array[halfSlice].y - min(raisingL, raisingR)
    imageSlice = sitkVolume[0:volumeSize[0], sliceSelected:sliceSelected+1, 0:volumeSize[2]]

    # Remove island
    for sliceNum in range(highestPoint.y +((v1Array[halfSlice].y - highestPoint.y) // 5), v1Array[halfSlice].y+(volumeSize[1] // 16)):
        res = sitkVolume[0:volumeSize[0], sliceNum:sliceNum+1, 0:volumeSize[2]]
        res = islandRemoval(res, volumeSize[0],  volumeSize[2])
        res.SetSpacing(imageSlice.GetSpacing())

        sitkVolume = sitk.Paste(destinationImage = sitkVolume,
                                sourceImage = res,
                                sourceSize = [volumeSize[0], 1, volumeSize[2]],
                                sourceIndex = [0, 0, 0],
                                destinationIndex = [0, sliceNum, 0])
    """---------------"""
    
    # Ensure the breast boundary be the circle shape
    breastBoundaryCircleShape = sitkVolume[0:volumeSize[0], circleShapeImageSliceSelected:circleShapeImageSliceSelected+1, 0:volumeSize[2]]
    
    sliceCount = volumeSize[1]
    for sliceNum in range(sliceSelected, sliceCount):
        nextSlice = sitkVolume[0:volumeSize[0], sliceNum:sliceNum+1, 0:volumeSize[2]]
        nextSlice.SetOrigin(imageSlice.GetOrigin())

        dilatedImage = sitk.BinaryDilate(imageSlice, (1, 1, 1), sitk.sitkBall, 0, 1, False)
        circleShapeDilateImage = sitk.BinaryDilate(breastBoundaryCircleShape, (8,8,8), sitk.sitkBall, 0, 1, False)
        circleShapeDilateImage.SetOrigin(imageSlice.GetOrigin())
        
        if sliceNum > v1Array[halfSlice].y + (volumeSize[1] // 8):
            mask = sitk.And(mask, 0)
        elif sliceNum > v1Array[halfSlice].y + (volumeSize[1] // 16):
            mask = sitk.And(circleShapeDilateImage, nextSlice)
        else:
            mask = sitk.And(dilatedImage, nextSlice)
        imageSlice = mask

        res = sitk.And(imageSlice, circleShapeDilateImage)

        sitkVolume = sitk.Paste(
            destinationImage = sitkVolume,
            sourceImage = res,
            sourceSize = [volumeSize[0], 1, volumeSize[2]],
            sourceIndex = [0, 0, 0],
            destinationIndex = [0, sliceNum, 0])

    ''' Operate in Coronal side '''

    outputImage = sitkVolume
    return outputImage

def islandRemoval(image, rows, cols):
    label = sitk.ConnectedComponent(image)
    labelArray = sitk.GetArrayFromImage(label)

    count_array = []
    for col in range(cols):
        count_array += list(labelArray[col][0])
    count = Counter(count_array)                        # Counter : get [(value1, count1), (value2, count2), (value3, count3) ...]

    if(len(count.most_common(3)) >= 2):
        if(count.most_common(3)[0][0] == 0):            # count.most_common(3) : get top3 (value,count) pairs that maybe including (0, count)
            one = count.most_common(3)[1][0]
            two = one
            if(len(count.most_common(3)) == 3):
                two = count.most_common(3)[2][0]
        elif(count.most_common(3)[1][0] == 0):
            one = count.most_common(3)[0][0]
            two = one
            if(len(count.most_common(3)) == 3):
                two = count.most_common(3)[2][0]
        else:
            one = count.most_common(3)[0][0]
            two = count.most_common(3)[1][0]
        for y in range(rows):
            for x in range(cols):
                com = labelArray[x][0][y]
                if(com != one and com != two):
                    labelArray[x][0][y] = 0
                else:
                    labelArray[x][0][y] = 1
    # Only 0 or 1
    image = sitk.GetImageFromArray(labelArray)          
    image = sitk.Cast(image, sitk.sitkUInt8)

    return image

def splitPectoralFromBreast(sliceNum, image, cutPointL, cutPointR):
    for i in range(cutPointL[sliceNum].x, cutPointR[sliceNum].x):
        for j in range(int(image.shape[1])):
            image[j][i] = 0

    return image

def findCutPoint(sliceNum, image, start, cutPointL, cutPointR, raisingL, raisingR):
    # Right -> Left
    for i in range(start.x,0,-1):
        if image[start.y - raisingL][i] > 0:
            cutPointL[sliceNum].x = i
            cutPointL[sliceNum].y = start.y - raisingL
            break

    # Left -> Right
    for i in range(start.x, int(image.shape[0]-1)):
        if image[start.y - raisingR][i] > 0:
            cutPointR[sliceNum].x = i
            cutPointR[sliceNum].y = start.y - raisingR
            break

def findOneHighestPoint(image):
    _y, _x = findHighestPoint(image)
    return _y
        
def findTwoHighestPoint(image, v1_x=0):
    if(v1_x > 0):
        middle = v1_x
    else:
        middle = int(image.shape[1])//2
        
    L_y, L_x = findHighestPoint(image, end_x = middle)
    R_y, R_x = findHighestPoint(image, start_x = middle)
                
    return (L_x,L_y), (R_x,R_y)

def findHighestPoint(image, start_x=0, end_x=0):
    if(end_x == 0 or end_x <= start_x):
        end_x = int(image.shape[1])
    
    for i in range(int(image.shape[0])):
        for j in range(start_x, end_x):
            if image[i][j] > 0:
                return i, j                          # high, width