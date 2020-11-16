import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import time
import sitkUtils
import SimpleITK as sitk
import numpy as np
import math

#
# STLModelBuilder
#

class STLModelBuilder(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "STLModelBuilder" # TODO make this more human readable by adding spaces
    self.parent.categories = ["NCTU"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# STLTestingWidget
#

class STLModelBuilderWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    pass

  def onApplyButton(self):
    logic = STLModelBuilderLogic()
    logic.run()
  
  def onReload(self):
    ScriptedLoadableModuleWidget.onReload(self)

#
# BatchTestingLogic
#

class STLModelBuilderLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def run(self):
    """
    Run the actual algorithm
    """

    logging.info('Processing started')

    """ 
    inputNodeNames = ['Patient 1', 'Patient 2', 'Patient 3', 'Patient 4',\
                      'Patient 5', 'Patient 6', 'Patient 7', 'Patient 8',\
                      'Patient 9', 'Patient 10', 'Patient 11', 'Patient 12',\
                      'Patient 13', 'Patient 14', 'Patient 15', 'Patient 16',\
                      'Patient 17', 'Patient 18', 'Patient 19', 'Patient 20',\
                      'Patient 21', 'Patient O30']

    iterationCounts = [4000, 6000, 1000, 8000,\
                        1000, 10000, 4000, 1000,\
                        1000, 10000, 8000, 8000,\
                        1000, 1000, 1000, 1000,\
                        8000, 8000, 1000, 10000,\
                        4000, 1000]
    """

    inputNodeNames =['Patient 1']
    iterationCounts = [4000]

    print "Start time: ", time.localtime(time.time())

    for i in range(len(inputNodeNames)):
      
      nodeName = inputNodeNames[i]
      inputNode = slicer.mrmlScene.GetFirstNodeByName(nodeName)
      if inputNode == None:
        continue

      logic.run(inputNode, None, None, iterationCounts[i])

    print "Complete time: ", time.localtime(time.time())
    logging.info('Processing completed')

    return True
  
  def customLogic(self, inputNode, outputNode):
    inputImage = sitkUtils.PullVolumeFromSlicer(inputNode)
    direction = inputImage.GetDirection()
    inputImage = sitk.Flip(inputImage, [direction[0] < 0, direction[4] < 0, direction[8] < 0])

    #inputImage = sitk.Normalize(inputImage)


    #value1Region = MRIBreastVolumeFunctions.PectoralSideModule.GetValue1Region(otsuMultiple)
    #edgeMap = MRIBreastVolumeFunctions.PectoralSideModule.GenerateEdgeMap(otsuMultiple)
    #outputImage = MRIBreastVolumeFunctions.PectoralSideModule.PruneEdgeMap(edgeMap, value1Region)
    #breastSideMask, v1, raisingL, raisingR = MRIBreastVolumeFunctions.BreastSideModule.EvaluateBreastSide(inputImage)

    #outputImage = MRIBreastVolumeFunctions.PectoralSideModule.EvaluatePectoralSide(inputImage)

    outputImage = sitk.Flip(outputImage, [direction[0] < 0, direction[4] < 0, direction[8] < 0])
    sitkUtils.PushVolumeToSlicer(outputImage, outputNode)
    return
    

    #outputImage = sitk.OtsuMultipleThresholds(inputImage, 2)
    outputImage = sitk.Multiply(inputImage, 2)
    outputImage = FillBody2(outputImage)
    outputImage = sitk.Greater(outputImage, 0)

    outputImage = sitk.Flip(outputImage, [direction[0] < 0, direction[4] < 0, direction[8] < 0])
    sitkUtils.PushVolumeToSlicer(outputImage, outputNode)
    return
    ###

    edgeMap = GenerateEdgeMap(inputImage)
    edgeList = BuildEdgeList(edgeMap)

    imageSize = inputImage.GetSize()
    depthMap = sitk.Image([imageSize[0], 1, imageSize[2]], sitk.sitkInt32)

    # Determine the center
    iCenter = int(math.floor(imageSize[0]/2))
    for k in range(imageSize[2]):
      if edgeList[iCenter][k].size > 0:
        depth = edgeList[iCenter][k][0]
      else:
        depth = -1
      depthMap.SetPixel([iCenter, 0, k], depth)
    InterpolateColumns(depthMap, iCenter)

    # 
    for i in range(iCenter + 1, imageSize[0]):
      for k in range(imageSize[2]):
        candidates = edgeList[i][k]

        if candidates.size == 0:
          depthMap.SetPixel([i, 0, k], -1)
          continue

        depth = -1
        """
        for candidate in candidates:
          bottomValid = k == 0 or 
        """
        squaredDiff = np.square(candidates - depthMap.GetPixel([i-1, 0, k]))
        if np.min(squaredDiff) > 9: depth = -1
        else : depth = candidates[np.argmin(squaredDiff)]
        depthMap.SetPixel([i, 0, k], depth)

      #InterpolateColumns(depthMap, i)

    outputImage = ReconstructFromDepthMap(depthMap, inputImage)

    outputImage = sitk.Flip(outputImage, [direction[0] < 0, direction[4] < 0, direction[8] < 0])
    sitkUtils.PushVolumeToSlicer(outputImage, outputNode)
    return

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

def FillBody(image):
  labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
  labelShapeStatistics.Execute(image)
  boundingBox = labelShapeStatistics.GetBoundingBox(1)
  pointOfTangencyJ = FindBackmostPointOfTangencyJ(image, boundingBox)
  whiteImage = sitk.Or(image, 1)
  closed = sitk.Paste(image1 = image,
                      image2 = whiteImage,
                      sourceSize = [boundingBox[3], boundingBox[4] - (pointOfTangencyJ - boundingBox[1]), boundingBox[5]],
                      sourceIndex = [boundingBox[0], pointOfTangencyJ, boundingBox[2]],
                      destinationIndex = [boundingBox[0], pointOfTangencyJ, boundingBox[2]])

  outputImage = sitk.Image(closed)
  imageSize = closed.GetSize()
  for k in range(imageSize[2]):
    slice = closed[0:imageSize[0], 0:imageSize[1], k]
    slice = sitk.BinaryFillhole(slice)
    outputImage = sitk.Paste(outputImage, sitk.JoinSeries(slice), [imageSize[0], imageSize[1], 1], [0, 0, 0], [0, 0, k])

  return outputImage

def FillBody2(otsuMultiple):
    imageSize = otsuMultiple.GetSize()

    ones = sitk.Image(imageSize, otsuMultiple.GetPixelID())
    ones.CopyInformation(otsuMultiple)
    ones = sitk.Add(ones, 1)

    for i in range(imageSize[0]):
        for k in range(imageSize[2]):
            for j in range(imageSize[1]-1, -1, -1):
                if otsuMultiple.GetPixel([i, j, k]) == 2:
                    otsuMultiple = sitk.Paste(image1 = otsuMultiple,
                                              image2 = ones,
                                              sourceSize = [1, imageSize[1] - j - 1, 1],
                                              sourceIndex = [i, j + 1, k],
                                              destinationIndex = [i, j + 1, k])
                    break

    binary = sitk.Greater(otsuMultiple, 0)
    binary = SlicewiseFillHole(binary)

    return sitk.Maximum(otsuMultiple, binary)

def FindBackmostPointOfTangencyJ(image, boundingBox):
  leftPointOfTangencyJ = rightPointOfTangencyJ = -1
  for j in range(boundingBox[1], boundingBox[1] + boundingBox[4]):
    for k in range(boundingBox[2], boundingBox[2] + boundingBox[5]):
      if leftPointOfTangencyJ == -1 and image.GetPixel([boundingBox[0], j, k]) > 0:
        leftPointOfTangencyJ = j
      if rightPointOfTangencyJ == -1 and image.GetPixel([boundingBox[0] + boundingBox[3] - 1, j, k]) > 0:
        rightPointOfTangencyJ = j
      if leftPointOfTangencyJ != -1 and rightPointOfTangencyJ != -1:
        return max(leftPointOfTangencyJ, rightPointOfTangencyJ)

def keepLargestComponent(image):
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

def DepthMap(image):
  imageSize = image.GetSize()
  outputImage = sitk.Image([imageSize[0], 1, imageSize[2]], sitk.sitkInt32)
  outputImage.SetOrigin(image.GetOrigin())
  outputImage.SetSpacing(image.GetSpacing())

  for i in range(imageSize[0]):
    for k in range(imageSize[2]):
      for j in range(imageSize[1]):
        if image.GetPixel([i, j, k]) > 0:
          outputImage.SetPixel([i, 0, k], imageSize[1] - j)
          break
  return outputImage

def BiasFieldCorrection2(image, mask):
    image = sitk.Cast(image, sitk.sitkFloat32)
    shrinkFactor = [4, 4, 1]
    shrinkedInput = sitk.Shrink(image, shrinkFactor)
    origin = shrinkedInput.GetOrigin()
    spacing = shrinkedInput.GetSpacing()
    direction = shrinkedInput.GetDirection()
    shrinkedInput.SetOrigin([0.0, 0.0, 0.0])
    shrinkedInput.SetSpacing([1.0, 1.0, 1.0])
    shrinkedInput.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    shrinkedMask = sitk.Shrink(mask, shrinkFactor)
    shrinkedMask.SetOrigin([0.0, 0.0, 0.0])
    shrinkedMask.SetSpacing([1.0, 1.0, 1.0])
    shrinkedMask.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    corrected = sitk.N4BiasFieldCorrection(
                    shrinkedInput,        # image
                    shrinkedMask,         # mask
                    0.001,                # convergenceThreshold
                    [50, 40, 30],         # maximumNumberOfIterations
                    0.4,                  # biasFieldFullWidthAtHalfMaximum
                    0.01,                 # wienerFilterNoise
                    200,                  # numberOfHistogramBins
                    [4, 4, 4],            # numberOfControlPoints
                    3,                    # splineOrder
                    False,                # useMaskLabel
                    )
    biasField = sitk.Subtract(corrected, shrinkedInput)
    biasField.SetOrigin(origin)
    biasField.SetSpacing(spacing)
    biasField.SetDirection(direction)
    biasField = sitk.Resample(biasField, image)
    return sitk.Add(image, biasField)

def BiasFieldCorrection(image):
    # Generate mask for N4ITK
    image = sitk.Cast(image, sitk.sitkFloat32)
    rescaled = sitk.RescaleIntensity(image, 0.0, 1.0)
    kmeans = sitk.ScalarImageKmeans(rescaled, [0.1, 0.3, 0.5, 0.7, 0.9])
    biasFieldCorrectionMask = sitk.Greater(kmeans, 0)

    # Create scene nodes
    inputNode = sitkUtils.PushVolumeToSlicer(image)
    maskNode = sitkUtils.PushVolumeToSlicer(biasFieldCorrectionMask)
    outputNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')

    # Run N4ITK CLI module
    n4itk = slicer.modules.n4itkbiasfieldcorrection
    parameters = {}
    parameters['inputImageName'] = inputNode.GetID()
    parameters['maskImageName'] = maskNode.GetID()
    parameters['outputImageName'] = outputNode.GetID()
    parameters['bfFWHM'] = 0.4
    slicer.cli.runSync(n4itk, None, parameters)

    # Retrieve output image
    outputImage = sitkUtils.PullVolumeFromSlicer(outputNode)

    # Clean up nodes
    slicer.mrmlScene.RemoveNode(inputNode)
    slicer.mrmlScene.RemoveNode(maskNode)
    slicer.mrmlScene.RemoveNode(outputNode)
    return outputImage

def GenerateEdgeMap(image):
  otsuMultiple = sitk.OtsuMultipleThresholds(image, 2)
  value2 = sitk.Equal(otsuMultiple, 2)

  kernel = sitk.Image([1, 3, 1], sitk.sitkInt8)
  kernel.SetPixel([0, 0, 0], 0)
  kernel.SetPixel([0, 1, 0], -1)
  kernel.SetPixel([0, 2, 0], 1)

  result = sitk.Convolution(sitk.Cast(value2, sitk.sitkInt8), kernel)
  result = sitk.Maximum(result, 0)
  return result

def BuildEdgeList(edgeMap):
  imageSize = edgeMap.GetSize()
  edgeList = [[[] for _ in range(imageSize[2])] for _ in range(imageSize[0])]
  for i in range(imageSize[0]):
    for k in range(imageSize[2]):
      edges = []
      for j in range(imageSize[1]):
        if edgeMap.GetPixel([i, j, k]) > 0:
          edges.append(j)
      edgeList[i][k] = np.array(edges)
  return edgeList

def FillHoles(image):
  padded = sitk.ConstantPad(image, [1, 0, 1], [1, 1, 1], 1)
  filled = sitk.BinaryFillhole(padded)
  filled = sitk.Crop(filled, [1, 0, 1], [1, 1, 1])
  return filled

def ForegroundMask(image):
  imageSize = image.GetSize()

  otsu = sitk.OtsuMultipleThresholds(image, 2)
  foreground = sitk.Greater(otsu, 0)

  for sliceIdx in range(imageSize[2]):
    imageSlice = keepLargestComponent(foreground[:, :, sliceIdx:sliceIdx + 1])
    foreground = sitk.Paste(image1 = foreground,
                            image2 = imageSlice,
                            sourceSize = [imageSize[0], imageSize[1], 1],
                            sourceIndex = [0, 0, 0],
                            destinationIndex = [0, 0, sliceIdx])
  return foreground

def RemoveValue1Islands(otsuImage):
  imageSize = otsuImage.GetSize()
  filled = sitk.Image(otsuImage)
  for sliceIdx in range(imageSize[2]):
    filledSlice = GrayscaleFillHole(otsuImage[:, :, sliceIdx:sliceIdx + 1])
    filled = sitk.Paste(image1 = filled,
                        image2 = filledSlice,
                        sourceSize = [imageSize[0], imageSize[1], 1],
                        sourceIndex = [0, 0, 0],
                        destinationIndex = [0, 0, sliceIdx])
  return filled

def GrayscaleFillHole(image):
  padded = sitk.ConstantPad(image, [1, 0, 1], [1, 1, 1], 2)
  filled = sitk.GrayscaleFillhole(padded)
  filled = sitk.Crop(filled, [1, 0, 1], [1, 1, 1])
  return filled

def RemoveValue2Islands(image):
  pass

def ReverseValue1AndValue2(otsuImage):
  value1 = sitk.Equal(otsuImage, 1)
  value2 = sitk.Equal(otsuImage, 2)
  reversed = sitk.Add(value2 , value1 * 2)
  return reversed

def ReconstructFromDepthMap(depthMap, refImage):
  imageSize = refImage.GetSize()
  output = sitk.Image(imageSize, refImage.GetPixelID())
  output.CopyInformation(refImage)

  for i in range(imageSize[0]):
    for k in range(imageSize[2]):
      depth = depthMap.GetPixel([i, 0, k])
      for j in range(0, depth):
        output.SetPixel([i, j, k], 1)
  return output

def CropAboveV1(foreground, v1):
  cropped = sitk.Image(foreground.GetSize(), sitk.sitkUInt8)
  cropped.CopyInformation(foreground)
  imageSize = cropped.GetSize()
  for sliceIdx in range(imageSize[2]):
    v1y = v1[sliceIdx].y
    cropped = sitk.Paste(image1 = cropped,
                        image2 = foreground,
                        sourceSize = [imageSize[0], imageSize[1] - v1y, 1],
                        sourceIndex = [0, v1y, sliceIdx],
                        destinationIndex = [0, v1y, sliceIdx])
  return cropped

def EstimateFatThickness(otsuMultiple, v1):
  imageSize = otsuMultiple.GetSize()
  for muscleDepth in range(imageSize[1]):
    if otsuMultiple.GetPixel([imageSize[0]/2, muscleDepth    , imageSize[2]/2]) == 2 and\
       otsuMultiple.GetPixel([imageSize[0]/2, muscleDepth + 1, imageSize[2]/2]) < 2:
      break
  fatThickness = muscleDepth - v1[imageSize[2]/2].y
  return fatThickness

def InterpolateColumns(depthImage, i):
  imageSize = depthImage.GetSize()

  kStart = 0
  while kStart < imageSize[2]:
    if depthImage[i, 0, kStart] < 0:

      kEnd = kStart + 1
      while kEnd < imageSize[2]:
        if depthImage[i, 0, kEnd] >= 0: break
        kEnd = kEnd + 1

      if kStart == 0 and kEnd == imageSize[2]:
        for k in range(kStart, kEnd):
          depthImage.SetPixel([i, 0, k], 0)

      elif kStart == 0:
        fillValue = depthImage.GetPixel([i, 0, kEnd])
        for k in range(kStart, kEnd):
          depthImage.SetPixel([i, 0, k], fillValue)

      elif kEnd == imageSize[2]:
        fillValue = depthImage.GetPixel([i, 0, kStart-1])
        for k in range(kStart, kEnd):
          depthImage.SetPixel([i, 0, k], fillValue)

      else:
        lerpStep = float(depthImage.GetPixel([i, 0, kEnd]) - depthImage.GetPixel([i, 0, kStart-1])) / (kEnd - kStart + 1)
        baseValue = depthImage.GetPixel([i, 0, kStart-1])
        for k in range(kStart, kEnd):
          depthImage.SetPixel([i, 0, k], int(round(baseValue + lerpStep * (k - kStart + 1))))
      
      kStart = kEnd

    kStart = kStart + 1

def SlicewiseFillHole(image):
    imageSize = image.GetSize()
    for k in range(imageSize[2]):
        axialSlice = image[0:imageSize[0], 0:imageSize[1], k:k+1]
        padded = sitk.ConstantPad(axialSlice, [0, 0, 1], [0, 1, 1], 1)
        filled = sitk.BinaryFillhole(padded)
        filled = sitk.Crop(filled, [0, 0, 1], [0, 1, 1])
        image = sitk.Paste(image1 = image,
                            image2 = filled,
                            sourceSize = [imageSize[0], imageSize[1], 1],
                            sourceIndex = [0, 0, 0],
                            destinationIndex = [0, 0, k])
    return image

def RoughCrop(value1):
    imageSize = value1.GetSize()
    cropped = sitk.Image(imageSize, value1.GetPixelID())
    cropped.CopyInformation(value1)

    for k in range(imageSize[2]):
        for j in range(1, imageSize[1]):
            if value1.GetPixel([imageSize[0]/2, j - 1, k]) == 0 and\
               value1.GetPixel([imageSize[0]/2, j    , k]) > 0:
                break

        cropOffsetInMm = 6
        cropOffset = int(round(cropOffsetInMm / value1.GetSpacing()[1]))
        cropJ = max(j - cropOffset, 0)

        cropped = sitk.Paste(image1 = cropped,
                            image2 = value1,
                            sourceSize = [imageSize[0], imageSize[1] - cropJ, imageSize[2]],
                            sourceIndex = [0, cropJ, 0],
                            destinationIndex = [0, cropJ, 0])
    return cropped

def SmoothByCurvatureFlow(binaryImg):
    levelSet = sitk.SignedDanielssonDistanceMap(binaryImg, True, False, True)
    levelSet = sitk.ZeroFluxNeumannPad(levelSet, [0, 0, 1], [0, 0, 1])

    potential = sitk.Image(levelSet.GetSize(), sitk.sitkFloat32)
    potential.CopyInformation(levelSet)
    potential = sitk.Add(potential, 1.0)

    levelSet = sitk.ShapeDetectionLevelSet(levelSet, potential, 0.0, 0.0, 1.0, 2000)
    levelSet = sitk.Crop(levelSet, [0, 0, 1], [0, 0, 1])

    return sitk.Greater(levelSet, 0)

class STLModelBuilderTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_STLModelBuilder1()

  def test_STLModelBuilder1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = STLModelBuilderLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
