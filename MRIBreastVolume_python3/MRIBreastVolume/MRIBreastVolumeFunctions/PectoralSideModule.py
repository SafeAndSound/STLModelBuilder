import slicer
import sitkUtils
import SimpleITK as sitk

def EvaluatePectoralSide(inputImage, smoothingIterations):
    smoothedImage = AnisotropicDiffusion(inputImage)
    otsuMultiple = sitk.OtsuMultipleThresholds(smoothedImage, 2)

    value1Region = GetValue1Region(otsuMultiple)
    edgeMap = GenerateEdgeMap(otsuMultiple)
    prunedEdgeMap = PruneEdgeMap(edgeMap, value1Region)

    roughPectoralShape = GetRoughPectoralShape(value1Region, smoothingIterations)
    pectoralShape = SnapToEdge(roughPectoralShape, prunedEdgeMap)
    return sitk.BinaryNot(pectoralShape)

def AnisotropicDiffusion(inputImage):
    smoothed = sitk.CurvatureAnisotropicDiffusion(
        image1 = inputImage,
        timeStep = 0.0625,
        conductanceParameter = 3.0,
        conductanceScalingUpdateInterval = 1,
        numberOfIterations = 8)
    return smoothed

def GetValue1Region(otsuMultiple):
    filled = FillBody(otsuMultiple)
    filled = RemoveSkin(filled)
    value1Region = sitk.Equal(filled, 1)
    value1Region = KeepLargestComponent(value1Region)
    value1Region = FillNarrowHoles(value1Region)
    return value1Region

def GetRoughPectoralShape(value1Region, smoothingIterations):
    cropped = RoughCrop(value1Region)
    shape = SmoothByCurvatureFlow(cropped, smoothingIterations)
    return shape

def FillBody(otsuMultiple):
    imageSize = otsuMultiple.GetSize()

    ones = sitk.Image(imageSize, otsuMultiple.GetPixelID())
    ones.CopyInformation(otsuMultiple)
    ones = sitk.Add(ones, 1)

    for i in range(imageSize[0]):
        for k in range(imageSize[2]):
            for j in range(imageSize[1]-1, -1, -1):
                if otsuMultiple.GetPixel([i, j, k]) == 2:
                    otsuMultiple = sitk.Paste(destinationImage = otsuMultiple,
                                              sourceImage = ones,
                                              sourceSize = [1, imageSize[1] - j - 1, 1],
                                              sourceIndex = [i, j + 1, k],
                                              destinationIndex = [i, j + 1, k])
                    break

    binary = sitk.Greater(otsuMultiple, 0)
    binary = SlicewiseFillHole(binary)

    return sitk.Maximum(otsuMultiple, binary)

def SlicewiseFillHole(image):
    imageSize = image.GetSize()
    for k in range(imageSize[2]):
        axialSlice = image[0:imageSize[0], 0:imageSize[1], k:k+1]
        padded = sitk.ConstantPad(axialSlice, [0, 0, 1], [0, 1, 1], 1)
        filled = sitk.BinaryFillhole(padded)
        filled = sitk.Crop(filled, [0, 0, 1], [0, 1, 1])
        image = sitk.Paste(destinationImage = image,
                            sourceImage = filled,
                            sourceSize = [imageSize[0], imageSize[1], 1],
                            sourceIndex = [0, 0, 0],
                            destinationIndex = [0, 0, k])
    return image

def RemoveSkin(otsuMultiple):
    binary = sitk.Greater(otsuMultiple, 0)
    imageSpacing = otsuMultiple.GetSpacing()
    erodeRadiusInMm = [1.5, 1.5, 0]
    erodeRadius = [int(round(erodeRadiusInMm[0] / imageSpacing[0])),\
                    int(round(erodeRadiusInMm[1] / imageSpacing[1])),\
                    int(round(erodeRadiusInMm[2] / imageSpacing[2]))]
    binary = sitk.BinaryErode(binary, erodeRadius, sitk.sitkBall)
    return sitk.Mask(otsuMultiple, binary)

def KeepLargestComponent(image):
    components = sitk.ConnectedComponent(image)
    labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelShapeStatistics.Execute(components)
    maximumLabel = 0
    maximumSize = 0
    for label in labelShapeStatistics.GetLabels():
        size = labelShapeStatistics.GetPhysicalSize(label)
        if size > maximumSize:
            maximumLabel = label
            maximumSize = size
    return sitk.Mask(image, sitk.Equal(components, maximumLabel))

def FillNarrowHoles(image):
    imageSpacing = image.GetSpacing()
    dilateRadiusInMm = [2.5, 2.5, 2.5]
    dilateRadius = [int(round(dilateRadiusInMm[0] / imageSpacing[0])),\
                    int(round(dilateRadiusInMm[1] / imageSpacing[1])),\
                    int(round(dilateRadiusInMm[2] / imageSpacing[2]))]
    dilated = sitk.BinaryDilate(image, dilateRadius, sitk.sitkBall)

    padded = sitk.ConstantPad(dilated, [0, 0, 1], [0, 0, 1], 1)
    filled = sitk.BinaryFillhole(padded)
    filled = sitk.Crop(filled, [0, 0, 1], [0, 0, 1])

    erodeRadiusInMm = [dilateRadiusInMm[0] * 2.0, dilateRadiusInMm[1] * 2.0, dilateRadiusInMm[2] * 2.0]
    erodeRadius = [int(round(erodeRadiusInMm[0] / imageSpacing[0])),\
                    int(round(erodeRadiusInMm[1] / imageSpacing[1])),\
                    int(round(erodeRadiusInMm[2] / imageSpacing[2]))]
    eroded = sitk.BinaryErode(filled, erodeRadius, sitk.sitkBall)
    return sitk.Or(image, eroded)

def RoughCrop(shape):
    imageSize = shape.GetSize()
    cropped = sitk.Image(imageSize, shape.GetPixelID())
    cropped.CopyInformation(shape)

    tmp = int(int(imageSize[0])/2)

    for k in range(imageSize[2]):
        for j in range(1, imageSize[1]):
            if shape.GetPixel([tmp, j - 1, k]) == 0 and\
               shape.GetPixel([tmp, j    , k]) > 0:
                break

        cropOffsetInMm = 6
        cropOffset = int(round(cropOffsetInMm / shape.GetSpacing()[1]))
        cropJ = max(j - cropOffset, 0)

        cropped = sitk.Paste(destinationImage = cropped,
                            sourceImage = shape,
                            sourceSize = [imageSize[0], imageSize[1] - cropJ, imageSize[2]],
                            sourceIndex = [0, cropJ, 0],
                            destinationIndex = [0, cropJ, 0])
    return cropped

def SmoothByCurvatureFlow(shape, iterations):
    levelSet = sitk.SignedDanielssonDistanceMap(shape, True, False, True)
    levelSet = sitk.ZeroFluxNeumannPad(levelSet, [0, 0, 1], [0, 0, 1])

    potential = sitk.Image(levelSet.GetSize(), sitk.sitkFloat32)
    potential.CopyInformation(levelSet)
    potential = sitk.Add(potential, 1.0)

    levelSet = sitk.ShapeDetectionLevelSet(levelSet, potential, 0.0, 0.0, 1.0, iterations)
    levelSet = sitk.Crop(levelSet, [0, 0, 1], [0, 0, 1])

    return sitk.Greater(levelSet, 0)

def GenerateEdgeMap(otsuMultiple):
  value2 = sitk.Equal(otsuMultiple, 2)

  kernel = sitk.Image([1, 3, 1], sitk.sitkInt8)
  kernel.SetPixel([0, 0, 0], 0)
  kernel.SetPixel([0, 1, 0], -1)
  kernel.SetPixel([0, 2, 0], 1)

  edgeMap = sitk.Convolution(sitk.Cast(value2, sitk.sitkInt8), kernel)
  edgeMap = sitk.Maximum(edgeMap, 0)
  return sitk.Cast(edgeMap, sitk.sitkUInt8)

def PruneEdgeMap(edgeMap, value1Region):
    imageSpacing = edgeMap.GetSpacing()
    erodeRadiusInMm = [2, 2, 0]
    erodeRadius = [int(round(erodeRadiusInMm[0] / imageSpacing[0])),\
                    int(round(erodeRadiusInMm[1] / imageSpacing[1])),\
                    int(round(erodeRadiusInMm[2] / imageSpacing[2]))]
    mask = sitk.BinaryErode(value1Region, erodeRadius, sitk.sitkBall)
    return sitk.MaskNegated(edgeMap, mask)

def SnapToEdge(pectoralShape, edgeMap):
    distanceMap = sitk.DanielssonDistanceMap(edgeMap, True, False, True)
    distanceThreshold = 3 # mm
    potential = sitk.Clamp(distanceMap, distanceMap.GetPixelID(), 0, distanceThreshold)
    potential = sitk.Divide(potential, distanceThreshold)
    potential = sitk.ZeroFluxNeumannPad(potential, [0, 0, 1], [0, 0, 1])

    levelSet = sitk.SignedDanielssonDistanceMap(pectoralShape, True, False, True)
    levelSet = sitk.ZeroFluxNeumannPad(levelSet, [0, 0, 1], [0, 0, 1])

    levelSet = sitk.GeodesicActiveContourLevelSet(levelSet, potential, 0.001, 0.0, 0.5, 1.0, 1500)
    levelSet = sitk.Crop(levelSet, [0, 0, 1], [0, 0, 1])

    return sitk.Greater(levelSet, 0)
