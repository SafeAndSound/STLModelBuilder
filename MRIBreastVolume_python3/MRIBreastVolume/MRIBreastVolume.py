import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import SimpleITK as sitk
import sitkUtils
import MRIBreastVolumeFunctions
import numpy as np
import MRIBreastVolumeFunctions.PectoralSideModule
import MRIBreastVolumeFunctions.BreastSideModule
import MRIBreastVolumeFunctions.SideBoundaryModule

#
# MRIBreastVolume
#

class MRIBreastVolume(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MRIBreastVolume" # TODO make this more human readable by adding spaces
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        self.parent.contributors = ["NCTU Computer Graphics Laboratory"] # replace with "Firstname Lastname (Organization)"
        self.parent.helpText = ""
        #self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = "" # replace with organization, grant and thanks.

#
# MRIBreastVolumeWidget
#

class MRIBreastVolumeWidget(ScriptedLoadableModuleWidget):
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
        parametersCollapsibleButton.setFont(qt.QFont("Times", 12))
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        #
        # input volume selector
        #
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene( slicer.mrmlScene )
        self.inputSelector.setToolTip( "Pick the input to the algorithm." )
        parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

        #
        # ROI selector
        #
        self.ROISelector = slicer.qMRMLNodeComboBox()
        self.ROISelector.nodeTypes = ["vtkMRMLAnnotationROINode"]
        self.ROISelector.selectNodeUponCreation = True
        self.ROISelector.addEnabled = False
        self.ROISelector.removeEnabled = True
        self.ROISelector.noneEnabled = True
        self.ROISelector.showHidden = False
        self.ROISelector.showChildNodeTypes = False
        self.ROISelector.setMRMLScene( slicer.mrmlScene )
        self.ROISelector.setToolTip( "Pick the ROI to the algorithm." )
        parametersFormLayout.addRow("Select ROI: ", self.ROISelector)

        #
        # Pectoral Smoothing Iterations Spin Box
        #
        self.pectoralSmoothingIterationSpinBox = qt.QSpinBox()
        self.pectoralSmoothingIterationSpinBox.setRange(0, 20000)
        self.pectoralSmoothingIterationSpinBox.setSingleStep(500)
        self.pectoralSmoothingIterationSpinBox.setValue(4000)
        parametersFormLayout.addRow("Pectoral Smoothing Iterations: ", self.pectoralSmoothingIterationSpinBox)

        #
        # Breast Implants Collapsible Button
        #
        breastImplantsCollapsibleButton = ctk.ctkCollapsibleButton()
        breastImplantsCollapsibleButton.text = "Breast Implants"
        breastImplantsCollapsibleButton.setFont(qt.QFont("Times", 12))
        breastImplantsCollapsibleButton.collapsed = True
        self.layout.addWidget(breastImplantsCollapsibleButton)

        # Layout within the dummy collapsible button
        breastImplantsFormLayout = qt.QFormLayout(breastImplantsCollapsibleButton)

        #
        # Breast implants algorithm hint label
        #
        self.breastImplantsHintLabel = qt.QLabel()
        self.breastImplantsHintLabel.setText("Please pick one point for one side or two points for both sides.")
        # Align Center
        self.breastImplantsHintLabel.setAlignment(4)
        self.breastImplantsHintLabel.setFrameStyle(qt.QFrame.WinPanel)
        self.breastImplantsHintLabel.setFrameShadow(qt.QFrame.Sunken)
        self.breastImplantsHintLabel.setMargin(2)
        self.breastImplantsHintLabel.setFont(qt.QFont("Times", 14, qt.QFont.Black))
        breastImplantsFormLayout.addRow(self.breastImplantsHintLabel)

        #
        # GACM initial point selector
        #
        self.pointSelector = slicer.qMRMLNodeComboBox()
        self.pointSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.pointSelector.selectNodeUponCreation = True
        self.pointSelector.addEnabled = True
        self.pointSelector.removeEnabled = True
        self.pointSelector.noneEnabled = True
        self.pointSelector.showHidden = False
        self.pointSelector.showChildNodeTypes = False
        self.pointSelector.baseName = "P"
        self.pointSelector.setMRMLScene( slicer.mrmlScene )
        self.pointSelector.setToolTip( "Pick the initial points for GACM." )
        breastImplantsFormLayout.addRow("Initial Points: ", self.pointSelector)

        #
        # Place the fiducial point(s) on screen
        #
        self.markupPointWidget = slicer.qSlicerMarkupsPlaceWidget()
        self.markupPointWidget.setMRMLScene(slicer.mrmlScene)
        self.markupPointWidget.setPlaceModePersistency(False)
        breastImplantsFormLayout.addRow(self.markupPointWidget)

        #
        # Estimate Volume Button
        #
        self.estimateButton = qt.QPushButton("Estimate Volume")
        self.estimateButton.toolTip = "Run the algorithm."
        self.estimateButton.enabled = False
        self.estimateButton.setFont(qt.QFont("Times", 24, qt.QFont.Black))
        self.layout.addWidget(self.estimateButton)

        #
        # Spacer
        #
        self.spacer = qt.QLabel()
        self.layout.addWidget(self.spacer)

        #
        # Editting Segmentation
        #
        segmentationEditorCollapsibleButton = ctk.ctkCollapsibleButton()
        segmentationEditorCollapsibleButton.text = "Editting Segmentation"
        segmentationEditorCollapsibleButton.setFont(qt.QFont("Times", 12))
        segmentationEditorCollapsibleButton.collapsed = True
        self.layout.addWidget(segmentationEditorCollapsibleButton)

        # Layout within the dummy collapsible button
        segmentationEditorFormLayout = qt.QFormLayout(segmentationEditorCollapsibleButton)

        self.segmentationEditorWidget = slicer.qMRMLSegmentEditorWidget()
        self.segmentationEditorWidget.setMaximumNumberOfUndoStates(10)
        self.parameterSetNode = None
        self.selectParameterNode()
        self.segmentationEditorWidget.setSwitchToSegmentationsButtonVisible(False)
        self.segmentationEditorWidget.setUndoEnabled(True)
        self.segmentationEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentationEditorFormLayout.addWidget(self.segmentationEditorWidget)

        #
        # Calculate Statistics Button
        #
        self.statButton = qt.QPushButton("Calculate Statistics")
        self.statButton.toolTip = "Calculate statistics."
        self.statButton.enabled = True
        self.statButton.setFont(qt.QFont("Times", 24, qt.QFont.Black))
        segmentationEditorFormLayout.addWidget(self.statButton)

        # connections
        self.estimateButton.connect('clicked(bool)', self.onEstimateButton)
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.pointSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.setPoint)
        self.segmentationEditorWidget.connect("masterVolumeNodeChanged(vtkMRMLVolumeNode*)", self.saveState)
        self.statButton.connect('clicked(bool)', self.calculateStatistics)

        # Add vertical spacer
        self.layout.addStretch(1)

        # Refresh Apply button state
        self.onSelect()

    def setPoint(self):
        self.markupPointWidget.setCurrentNode(self.pointSelector.currentNode())

    def cleanup(self):
        pass

    def onSelect(self):
        self.estimateButton.enabled = True

    def onEstimateButton(self):
        logic = MRIBreastVolumeLogic()
        logic.Run(self.inputSelector.currentNode(), self.ROISelector.currentNode(), self.pointSelector.currentNode(), self.pectoralSmoothingIterationSpinBox.value)

    def onReload(self):
        reload(MRIBreastVolumeFunctions.PectoralSideModule)
        reload(MRIBreastVolumeFunctions.BreastSideModule)
        reload(MRIBreastVolumeFunctions.SideBoundaryModule)
        ScriptedLoadableModuleWidget.onReload(self)

    def calculateStatistics(self):
        from SegmentStatistics import SegmentStatisticsLogic
        segStatLogic = SegmentStatisticsLogic()

        segStatLogic.getParameterNode().SetParameter("Segmentation", self.segmentationEditorWidget.segmentationNodeID())
        self.segmentationEditorWidget.segmentationNode().CreateDefaultDisplayNodes()
        segStatLogic.computeStatistics()

        resultsTableNode = slicer.vtkMRMLTableNode()
        slicer.mrmlScene.AddNode(resultsTableNode)
        segStatLogic.exportToTable(resultsTableNode)
        segStatLogic.showTable(resultsTableNode)

    def selectParameterNode(self):
        # Select parameter set node if one is found in the scene, and create one otherwise
        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        if self.parameterSetNode == segmentEditorNode:
            # nothing changed
            return
        self.parameterSetNode = segmentEditorNode
        self.segmentationEditorWidget.setMRMLSegmentEditorNode(self.parameterSetNode)

    def saveState(self):
        self.segmentationEditorWidget.saveStateForUndo()

#
# MRIBreastVolumeLogic
#

class MRIBreastVolumeLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def HasImageData(self,volumeNode):
        """This is an example logic method that
        returns true if the passed in volume
        node has valid image data
        """
        if not volumeNode:
            logging.debug('HasImageData failed: no volume node')
            return False
        if volumeNode.GetImageData() is None:
            logging.debug('HasImageData failed: no image data in volume node')
            return False
        return True

    # Actual algorithm
    def Run(self, inputVolume, ROI=None, initPoint=None, pectoralSmoothingIterations=4000):

        logging.info('Processing started')
        inputImage = sitkUtils.PullVolumeFromSlicer(inputVolume)
        direction = inputImage.GetDirection()
        inputImage = sitk.Flip(inputImage, [direction[0] < 0, direction[4] < 0, direction[8] < 0])

        if initPoint != None:
            fiducialCoord = self.getFiducialCoord(inputVolume, initPoint)                                   # (inputVolume, initPoint)
            msg = self.foolProof(inputImage, fiducialCoord)
            if(sum(msg)):
                self.showErrorMSG(msg)
                return False

        inputImage = self.BiasFieldCorrection(inputImage)
        pectoralSideMask = MRIBreastVolumeFunctions.PectoralSideModule.EvaluatePectoralSide(inputImage, pectoralSmoothingIterations)
        breastSideMask, v1, raisingL, raisingR = MRIBreastVolumeFunctions.BreastSideModule.EvaluateBreastSide(inputImage, sitk.BinaryNot(pectoralSideMask))
        sideBoundaryMask = MRIBreastVolumeFunctions.SideBoundaryModule.EvaluateSideBoundary(breastSideMask, v1, raisingL, raisingR)
        mask = sitk.And(sideBoundaryMask, pectoralSideMask)

        if ROI != None:
            mask = self.cropByROI(mask, ROI)

        if initPoint != None:
            implantsVolume =  self.getBreastImplantsVolume(inputImage, fiducialCoord)
            mask = sitk.And(mask, sitk.BinaryNot(implantsVolume))

        # Create segmentation Node
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName(inputVolume.GetName()+'_seg')

        # Split into left and right part
        leftBreast, rightBreast = self.splitBreasts(mask, v1)
        if initPoint != None:
            leftImplant, rightImplant = self.splitBreasts(implantsVolume, v1)

        # Restore the direction
        leftBreast = sitk.Flip(leftBreast, [direction[0] < 0, direction[4] < 0, direction[8] < 0])
        rightBreast = sitk.Flip(rightBreast, [direction[0] < 0, direction[4] < 0, direction[8] < 0])
        if initPoint != None:
            leftImplant = sitk.Flip(leftImplant, [direction[0] < 0, direction[4] < 0, direction[8] < 0])
            rightImplant = sitk.Flip(rightImplant, [direction[0] < 0, direction[4] < 0, direction[8] < 0])

        # Create VtkOrientedImage
        vtkLeftBreast = self.sitkImageToVtkOrientedImage(leftBreast)
        vtkRightBreast = self.sitkImageToVtkOrientedImage(rightBreast)
        if initPoint != None:
            vtkLeftImplant = self.sitkImageToVtkOrientedImage(leftImplant)
            vtkRightImplant = self.sitkImageToVtkOrientedImage(rightImplant)

        # Add segmentation
        segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(vtkLeftBreast, "LeftBreast", [1.0, 1.0, 0.0])
        segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(vtkRightBreast, "RightBreast", [0.0, 0.0, 1.0])
        if initPoint != None:
            segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(vtkLeftImplant, "LeftImplant", [1.0, 0.5, 0.0])
            segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(vtkRightImplant, "RightImplant", [0.0, 1.0, 1.0])

        logging.info('Processing completed')
        return True

    def BiasFieldCorrection(self, image):
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

    def sitkImageToVtkOrientedImage(self, img):
        imgNode = sitkUtils.PushVolumeToSlicer(img)
        vtkImage = imgNode.GetImageData()

        vtkOrientedImage = slicer.vtkOrientedImageData()
        vtkOrientedImage.DeepCopy(vtkImage)
        dir = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
        imgNode.GetIJKToRASDirections(dir)
        vtkOrientedImage.SetDirections([dir[0], dir[1], dir[2]])
        vtkOrientedImage.SetOrigin(imgNode.GetOrigin())
        vtkOrientedImage.SetSpacing(imgNode.GetSpacing())

        slicer.mrmlScene.RemoveNode(imgNode)
        return vtkOrientedImage

    def splitBreasts(self, image, v1):
        imageSize = image.GetSize()
        left = sitk.Image(imageSize, sitk.sitkUInt8)
        left.CopyInformation(image)
        right = sitk.Image(imageSize, sitk.sitkUInt8)
        right.CopyInformation(image)

        splitThreshold = v1[int(imageSize[2] * 0.5)].x

        components = sitk.ConnectedComponent(image)
        labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
        labelShapeStatistics.Execute(components)
        for label in labelShapeStatistics.GetLabels():
            centroid = labelShapeStatistics.GetCentroid(label)
            centroid = image.TransformPhysicalPointToIndex(centroid)
            if centroid[0] < splitThreshold:
                right = sitk.Or(right, sitk.Equal(components, label))
            else:
                left = sitk.Or(left, sitk.Equal(components, label))

        return left, right

    def cropByROI(self, mask, ROI):
        roiBound = np.zeros(6)

        # Return (Xmin, Xmax, Ymin, Ymax, Zmin, Zmax)
        ROI.GetRASBounds(roiBound)

        # RAS coordinate to LPS coordinate
        minXYZ = mask.TransformPhysicalPointToIndex([-roiBound[1], -roiBound[3], roiBound[5]])
        maxXYZ = mask.TransformPhysicalPointToIndex([-roiBound[0], -roiBound[2], roiBound[4]])

        breastVolume = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
        breastVolume.CopyInformation(mask)

        bbox_width = maxXYZ[0] - minXYZ[0]
        bbox_height = maxXYZ[1] - minXYZ[1]

        breastVolume = sitk.Paste(
            image1 = breastVolume,
            image2 = mask,
            sourceSize = [bbox_width, bbox_height, mask.GetDepth()],
            sourceIndex = [minXYZ[0], minXYZ[1], 0],
            destinationIndex = [minXYZ[0], minXYZ[1], 0]
        )

        return breastVolume

    def getFiducialCoord(self, input, point):
        dir = vtk.vtkMatrix4x4()
        input.GetRASToIJKMatrix(dir)
        coord = np.zeros(4)
        coord_ = []

        n = point.GetNumberOfFiducials()
        for i in range(n):
            point.GetNthFiducialWorldCoordinates(i, coord)
            coord_.append((int(dir.MultiplyPoint(coord)[0]),int(dir.MultiplyPoint(coord)[1]),int(dir.MultiplyPoint(coord)[2])))
        
        point.RemoveAllMarkups()
        return coord_

    def getBreastImplantsVolume(self, sitkVolume, fiducialCoord):
        sitkVolume = sitk.Normalize(sitkVolume)

        # measure implants's group based on fiducial points. (dark or bright)
        group = self.detectGroup(sitkVolume, fiducialCoord)
        
        # compute gradient based on implants's group
        EdgeImage = self.computeGradientWithPreprocessing(sitkVolume, group)

        # GACM
        levelSet = self.findBreastImplantsRegion(EdgeImage, fiducialCoord)

        # remain negative region
        levelSet = sitk.BinaryThreshold(
            image1 = levelSet,
            lowerThreshold = -100.0,
            upperThreshold = 0.0,
            insideValue = 1,
            outsideValue = 0)

        if(group > 1):
            levelSet = sitk.BinaryDilate(levelSet, (1,1,0), sitk.sitkBall)
        return levelSet

    def detectGroup(self, sitkVolume, fiducialCoord):
        otsu = sitk.OtsuMultipleThresholds(sitkVolume, numberOfThresholds=2)                           # 0(dark) or 1 or 2(bright)
        group = 0.0
        numberOfPoints = 1
        k = 0
        for k in range(len(fiducialCoord)):
            for i in range(-2,2,1): # kernel size = 4*4
                for j in range(-2,2,1):
                        group += otsu[fiducialCoord[k][0]+i,fiducialCoord[k][1]+j,fiducialCoord[k][2]]         # another point

        numberOfPoints += k
        group = round(float(group) / (4*4*numberOfPoints))

        return group

    def computeGradientWithPreprocessing(self, sitkVolume, group):
        # Set parameters for group 2 (brighter) or group 01 (darker).
        # Group 01 need to raise edge sensitive. On the contrary, group 2 need to reduce edge sensitive.
        if(group > 1.0):
            alpha = -0.1
            beta = 0.3
            iteration = 7
            kernelSize = (3,3,2)
        else:
            alpha = -0.1
            beta = 0.0
            iteration = 3
            kernelSize = (4,4,2)

        EdgeImage = sitk.CurvatureAnisotropicDiffusion(sitkVolume, 
                                        timeStep = 0.0625,
                                        conductanceParameter = 3.0,
                                        conductanceScalingUpdateInterval = 1,
                                        numberOfIterations = iteration)

        EdgeImage = sitk.LaplacianSharpening(EdgeImage)                                                 # Edge enhancement
            
        EdgeImage = sitk.GradientMagnitude(EdgeImage)                                                   # Computes the gradient magnitude of an image region at each pixel.

        EdgeImage = sitk.GrayscaleMorphologicalClosing(EdgeImage, kernelSize, sitk.sitkBall, True)      # Let the region of implants closed.

        EdgeImage = sitk.Sigmoid(EdgeImage,
                            alpha = alpha,
                            beta = beta,
                            outputMaximum = 1.0,
                            outputMinimum = 0.0)

        return EdgeImage

    def findBreastImplantsRegion(self, EdgeImage, fiducialCoord):
        # The FastMarch measure distance on a map, that is initial levelSet.
        levelSet = sitk.FastMarching(EdgeImage,
                        trialPoints=tuple(fiducialCoord),
                        normalizationFactor=0.5,
                        stoppingValue=10000)
        
        # In initial, the levelSet should have negative value. e.g. 0 ~ 10000 -> -50 ~ 9950
        levelSet = sitk.Minimum(levelSet, 10000.0)
        levelSet = sitk.Cast(levelSet, sitk.sitkFloat32)
        levelSet = sitk.Add(levelSet, -50.0)

        # move levelSet, this is GACM.
        levelSet = sitk.GeodesicActiveContourLevelSet(
                levelSet,
                EdgeImage,
                propagationScaling = 1.0,
                curvatureScaling = 0.0,
                advectionScaling = 0.0,
                numberOfIterations = 1000,
                reverseExpansionDirection = False)
        
        return levelSet

    def foolProof(self, inputVolume, fiducialCoord):
        normalizeVolume = sitk.Normalize(inputVolume)
        isFool1 = self.lessThan3Points(fiducialCoord)
        isFool2 = self.pointCanNotBeSameSide(normalizeVolume, fiducialCoord)
        isFool3 = self.implantsBoundingBox(normalizeVolume, fiducialCoord)
        
        msg = (isFool1, isFool2, isFool3)
        return msg

    def showErrorMSG(self, msg):
        errorDisplay = ['Pick Too Many Points(<=2)!']
        errorDisplay += ['Points On The Same Side!']
        errorDisplay += ['Points Outside Implant Region!']

        for i in range(len(msg)):
            if(msg[i]):
                slicer.util.errorDisplay(errorDisplay[i])
                return

    def lessThan3Points(self, fiducialCoord):
        if(len(fiducialCoord) > 2):
            return True
        else:
            return False

    def pointCanNotBeSameSide(self, inputVolume, fiducialCoord):
        volumeSize = list(inputVolume.GetSize())                                                                 # Get volume size
        middle = volumeSize[0]//2
        fool = False
        if(len(fiducialCoord) > 1):
            if(fiducialCoord[0][0] < middle and fiducialCoord[1][0] < middle):
                fool = True
            elif(fiducialCoord[0][0] > middle and fiducialCoord[1][0] > middle):
                fool = True
        return fool

    def implantsBoundingBox(self, inputVolume, fiducialCoord):
        volumeSize = list(inputVolume.GetSize())                                                                 # Get volume size
        imageSlice = inputVolume[0:volumeSize[0], 0:volumeSize[1], volumeSize[2]//2]
        binaryImage = MRIBreastVolumeFunctions.BreastSideModule.preprocess(imageSlice, pixelSpacing_y=inputVolume.GetSpacing()[1])
        image = sitk.GetArrayFromImage(binaryImage)

        self.v1_x, self.v1_y = MRIBreastVolumeFunctions.BreastSideModule.findSingleV1(image)
        self.L, self.R = MRIBreastVolumeFunctions.SideBoundaryModule.findTwoHighestPoint(image, self.v1_x)

        distance_x = max(abs(self.v1_x-self.L[0]),abs(self.v1_x-self.R[0]))
        distance_y = max(abs(self.v1_y-self.L[1]),abs(self.v1_y-self.R[1]))
        for i in range(len(fiducialCoord)):
            if not (abs(fiducialCoord[i][0]-self.L[0]) <= distance_x or abs(fiducialCoord[i][0]-self.R[0]) <= distance_x):
                return True
            if not (abs(fiducialCoord[i][1]-self.v1_y) <= distance_y):
                return True
        return False

class MRIBreastVolumeTest(ScriptedLoadableModuleTest):
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
        self.test_MRIBreastVolume1()

    def test_MRIBreastVolume1(self):
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
        logic = MRIBreastVolumeLogic()
        self.assertIsNotNone( logic.HasImageData(volumeNode) )
        self.delayDisplay('Test passed!')
