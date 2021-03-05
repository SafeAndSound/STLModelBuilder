import logging
import math
import os
import random
import sys
import time
import unittest

import ctk
import numpy as np
import qt
import SimpleITK as sitk
import sitkUtils
import slicer
import vtk
from slicer.ScriptedLoadableModule import *
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

#
# STLModelBuilder
#


class STLModelBuilder(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        # TODO make this more human readable by adding spaces
        self.parent.title = "STLModelBuilder"
        self.parent.categories = ["NCTU"]
        self.parent.dependencies = []
        # replace with "Firstname Lastname (Organization)"
        self.parent.contributors = ["NCTU CG Lab"]
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""  # replace with organization, grant and thanks.

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
        # input volume selector
        #
        self.inputModelSelector = slicer.qMRMLNodeComboBox()
        self.inputModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.inputModelSelector.selectNodeUponCreation = True
        self.inputModelSelector.addEnabled = False
        self.inputModelSelector.removeEnabled = False
        self.inputModelSelector.noneEnabled = False
        self.inputModelSelector.showHidden = False
        self.inputModelSelector.showChildNodeTypes = False
        self.inputModelSelector.setMRMLScene( slicer.mrmlScene )
        self.inputModelSelector.setToolTip( "Model node containing geometry and texture coordinates." )
        parametersFormLayout.addRow("Input OBJ Model: ", self.inputModelSelector)

        #input texture selector
        self.inputTextureSelector = slicer.qMRMLNodeComboBox()
        self.inputTextureSelector.nodeTypes = [ "vtkMRMLVectorVolumeNode" ]
        self.inputTextureSelector.addEnabled = False
        self.inputTextureSelector.removeEnabled = False
        self.inputTextureSelector.noneEnabled = False
        self.inputTextureSelector.showHidden = False
        self.inputTextureSelector.showChildNodeTypes = False
        self.inputTextureSelector.setMRMLScene( slicer.mrmlScene )
        self.inputTextureSelector.setToolTip( "Color image containing texture image." )
        parametersFormLayout.addRow("Texture: ", self.inputTextureSelector)

        # inpute color selector
        self.targetColor = qt.QColor("#4573a0")
        self.colorButton = qt.QPushButton()
        self.colorButton.setStyleSheet("background-color: " + self.targetColor.name())
        parametersFormLayout.addRow("Marker Color:", self.colorButton)

        # Texture Button
        self.textureButton = qt.QPushButton("Apply Texture")
        self.textureButton.toolTip = "Paste the texture onto the model."
        self.textureButton.enabled = False
        parametersFormLayout.addRow(self.textureButton)

        # Apply Button
        self.applyButton = qt.QPushButton("Remove Tape")
        self.applyButton.toolTip = "Run the algorithm."
        self.applyButton.enabled = False
        parametersFormLayout.addRow(self.applyButton)

        # Select Breats
        self.breatButton = qt.QPushButton("Finish Select Breats")
        self.breatButton.toolTip = "Click after breats are selected."
        self.breatButton.enabled = False
        parametersFormLayout.addRow(self.breatButton)

        # connections
        self.colorButton.connect('clicked(bool)', self.onSelectColor)
        self.textureButton.connect('clicked(bool)', self.onTextureButton)
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.breatButton.connect('clicked(bool)', self.onBreatButton)
        self.inputModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelectInputData)
        self.inputTextureSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelectInputData)

        self.onSelectInputData()

        # Add vertical spacer
        self.layout.addStretch(1)

        self.logic = STLModelBuilderLogic()
        self.logic.initiate(self)

    def cleanup(self):
        pass

    def onSelectInputData(self):
        self.textureButton.enabled = self.inputTextureSelector.currentNode() and self.inputModelSelector.currentNode()
        self.applyButton.enabled = self.inputTextureSelector.currentNode() and self.inputModelSelector.currentNode()

    def onSelectColor(self):
        self.targetColor = qt.QColorDialog.getColor()
        self.colorButton.setStyleSheet("background-color: " + self.targetColor.name())
        self.colorButton.update()
    
    def onTextureButton(self):
        self.logic.showTextureOnModel(self.inputModelSelector.currentNode(), self.inputTextureSelector.currentNode())

    def onApplyButton(self):
        self.logic.run(self.inputModelSelector.currentNode(), self.inputTextureSelector.currentNode(), self.targetColor)

    def onBreatButton(self):
        self.logic.truncateBreastPolyData("Reference_Breast_Position")

    def finishPreProcessing(self):
        self.breatButton.enabled = True
        self.logic.setupFiducialNodeOperation("Reference_Breast_Position")
        
    def onReload(self):
        ScriptedLoadableModuleWidget.onReload(self)

class STLModelBuilderLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def initiate(self, widget):
        self.widget = widget
        self.modifidedModelNode = None
    
    def showTextureOnModel(self, modelNode, textureImageNode):
        modelDisplayNode = modelNode.GetDisplayNode()
        modelDisplayNode.SetBackfaceCulling(0)
        textureImageFlipVert = vtk.vtkImageFlip()
        textureImageFlipVert.SetFilteredAxis(1)
        textureImageFlipVert.SetInputConnection(textureImageNode.GetImageDataConnection())
        modelDisplayNode.SetTextureImageDataConnection(textureImageFlipVert.GetOutputPort())
    
    def convertTextureToPointAttribute(self, modelNode, textureImageNode):
        polyData = modelNode.GetPolyData()
        textureImageFlipVert = vtk.vtkImageFlip()
        textureImageFlipVert.SetFilteredAxis(1)
        textureImageFlipVert.SetInputConnection(textureImageNode.GetImageDataConnection())
        textureImageFlipVert.Update()
        textureImageData = textureImageFlipVert.GetOutput()
        pointData = polyData.GetPointData()
        tcoords = pointData.GetTCoords()
        numOfPoints = pointData.GetNumberOfTuples()
        assert numOfPoints == tcoords.GetNumberOfTuples(), "Number of texture coordinates does not equal number of points"
        textureSamplingPointsUv = vtk.vtkPoints()
        textureSamplingPointsUv.SetNumberOfPoints(numOfPoints)
        for pointIndex in range(numOfPoints):
            uv = tcoords.GetTuple2(pointIndex)
            textureSamplingPointsUv.SetPoint(pointIndex, uv[0], uv[1], 0)

        textureSamplingPointDataUv = vtk.vtkPolyData()
        uvToXyz = vtk.vtkTransform()
        textureImageDataSpacingSpacing = textureImageData.GetSpacing()
        textureImageDataSpacingOrigin = textureImageData.GetOrigin()
        textureImageDataSpacingDimensions = textureImageData.GetDimensions()
        uvToXyz.Scale(textureImageDataSpacingDimensions[0] / textureImageDataSpacingSpacing[0],
                  textureImageDataSpacingDimensions[1] / textureImageDataSpacingSpacing[1], 1)
        uvToXyz.Translate(textureImageDataSpacingOrigin)
        textureSamplingPointDataUv.SetPoints(textureSamplingPointsUv)
        transformPolyDataToXyz = vtk.vtkTransformPolyDataFilter()
        transformPolyDataToXyz.SetInputData(textureSamplingPointDataUv)
        transformPolyDataToXyz.SetTransform(uvToXyz)
        probeFilter = vtk.vtkProbeFilter()
        probeFilter.SetInputConnection(transformPolyDataToXyz.GetOutputPort())
        probeFilter.SetSourceData(textureImageData)
        probeFilter.Update()
        rgbPoints = probeFilter.GetOutput().GetPointData().GetArray('ImageScalars')

        colorArray = vtk.vtkDoubleArray()
        colorArray.SetName('Color')
        colorArray.SetNumberOfComponents(3)
        colorArray.SetNumberOfTuples(numOfPoints)
        for pointIndex in range(numOfPoints):
            rgb = rgbPoints.GetTuple3(pointIndex)
            colorArray.SetTuple3(pointIndex, rgb[0]/255., rgb[1]/255., rgb[2]/255.)
            colorArray.Modified()
            pointData.AddArray(colorArray)

        pointData.Modified()
        polyData.Modified()

    def run(self, modelNode, textureImageNode, targetColor):
        """
        Run the actual algorithm
        """
        print("----Start Processing----")
        startTime = time.time()
        print("Start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(startTime)) + "\n")

        newPolyData = vtk.vtkPolyData()
        newPolyData.DeepCopy(modelNode.GetPolyData())
        self.modifidedModelNode = self.createNewModelNode(newPolyData, "Modified_Model")
        newModelNode = self.modifidedModelNode

        #轉換顏色格式:QColor -> np.array
        targetColor = np.array([targetColor.redF(), targetColor.greenF(), targetColor.blueF()])
        print("Selected Color: {}".format(targetColor))

        #取得vtkMRMLModelNode讀取的檔案
        fileName = modelNode.GetStorageNode().GetFileName()
        print("OBJ File Path: {}\n".format(fileName))

        print("Origin Model points: {}".format(self.modifidedModelNode.GetPolyData().GetNumberOfPoints()))
        print("Origin Model cells: {}\n".format(self.modifidedModelNode.GetPolyData().GetNumberOfCells()))

        #產生點的顏色資料
        self.convertTextureToPointAttribute(newModelNode, textureImageNode)

        #取出顏色於範圍內的點id
        delPointIds = self.extractSelection(newModelNode, targetColor, 0.16)

        #刪除顏色符合的點
        newModelNode.SetAndObservePolyData(self.deletePoint(newModelNode.GetPolyData(), delPointIds))

        #處理PolyData (降低面數、破洞處理......)
        self.reduceAndCleanPolyData(newModelNode)

        print("Modified Model points: {}".format(newModelNode.GetPolyData().GetNumberOfPoints()))
        print("Modified Model cells: {}\n".format(newModelNode.GetPolyData().GetNumberOfCells()))

        modelNode.GetDisplayNode().VisibilityOff()
        newModelNode.GetDisplayNode().VisibilityOn()

        self.widget.finishPreProcessing()

        print("\n----Complete Processing----")
        stopTime = time.time()
        print("Complete time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stopTime)))
        logging.info('Processing completed in {0:.2f} seconds\n'.format(stopTime - startTime))

        return True
    
    def reduceAndCleanPolyData(self, modelNode):
        #triangulate
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(modelNode.GetPolyData())
        triangleFilter.Update()

        #decimate
        decimateFilter = vtk.vtkDecimatePro()
        decimateFilter.SetInputConnection(triangleFilter.GetOutputPort())
        decimateFilter.SetTargetReduction(0.33)
        decimateFilter.PreserveTopologyOn()
        decimateFilter.BoundaryVertexDeletionOff()
        decimateFilter.Update()

        #clean
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(decimateFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        #relax
        relaxFilter = vtk.vtkWindowedSincPolyDataFilter()
        relaxFilter.SetInputConnection(cleanFilter.GetOutputPort())
        relaxFilter.SetNumberOfIterations(10)
        relaxFilter.BoundarySmoothingOn()
        relaxFilter.FeatureEdgeSmoothingOn()
        relaxFilter.SetFeatureAngle(120.0)
        relaxFilter.SetPassBand(0.001)
        relaxFilter.NonManifoldSmoothingOn()
        relaxFilter.NormalizeCoordinatesOn()
        relaxFilter.Update()

        #normal
        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputConnection(relaxFilter.GetOutputPort())
        normalFilter.ComputePointNormalsOn()
        normalFilter.SplittingOff()
        normalFilter.Update()

        #alignCenter
        polyData = normalFilter.GetOutput()
        points_array = vtk_to_numpy(polyData.GetPoints().GetData())
        center = points_array.sum(axis = 0) / points_array.shape[0]
        np.copyto(points_array, points_array - center)
        polyData.GetPoints().GetData().Modified()

        modelNode.SetAndObservePolyData(polyData)

    def extractSelection(self, modelNode, targetColor, threshold):
        colorData = vtk_to_numpy(modelNode.GetPolyData().GetPointData().GetArray("Color"))
        colorData = np.sum(np.abs(colorData - targetColor), axis=1) / 3

        return np.asarray(np.where(colorData < threshold))[0]

    def deletePoint(self, polyData, delPointIds):
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(numpy_to_vtk(delPointIds))
        selectionNode.GetProperties().Set(vtk.vtkSelectionNode.INVERSE(), 1)
        selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, polyData)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()

        geometryFilter = vtk.vtkGeometryFilter()
        geometryFilter.SetInputData(extractSelection.GetOutput())
        geometryFilter.Update()

        return geometryFilter.GetOutput()

    def createNewModelNode(self, polyData, nodeName):
        modelNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLModelNode())
        modelNode.SetName(nodeName)
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(polyData)

        return modelNode

    def setupFiducialNodeOperation(self, nodeName):
        #Create fiducial node
        fiducialNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLMarkupsFiducialNode())
        fiducialNode.SetName(nodeName)

        placeModePersistence = 1
        slicer.modules.markups.logic().StartPlaceMode(placeModePersistence)

    def truncateBreastPolyData(self, nodeName):
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SwitchToViewTransformMode()
        # also turn off place mode persistence if required
        interactionNode.SetPlaceModePersistence(0)

        fiducialNode = slicer.util.getNode(nodeName)
        numFids = fiducialNode.GetNumberOfFiducials()

        breastPos = []
        # Use the last 2 fiducials as two breast positions
        for i in range(numFids - 2, numFids):
            ras = [0,0,0]
            fiducialNode.GetNthFiducialPosition(i, ras)
            breastPos.append(ras)
            # the world position is the RAS position with any transform matrices applied
            world = [0,0,0,0]
            fiducialNode.GetNthFiducialWorldCoordinates(i, world)
            #print(i, ": RAS =", ras, ", world =", world)
        
        slicer.mrmlScene.RemoveNode(fiducialNode)

        for i in range(2):
            #尋找最接近選取點的mesh
            connectFilter = vtk.vtkPolyDataConnectivityFilter()
            connectFilter.SetInputData(self.modifidedModelNode.GetPolyData())
            connectFilter.SetExtractionModeToClosestPointRegion()
            connectFilter.SetClosestPoint(breastPos[i])
            connectFilter.Update()

            refinedBreastPolyData = self.refineBreastPolyData(connectFilter.GetOutput(), 10)

            nonEdgePolyData = refinedBreastPolyData
            for j in range(3): #藉由直接移除n層boundary減少突出邊緣
                _, edgeIds = self.extractBoundaryPoints(nonEdgePolyData, "Edge_{}_Smooth_{}".format(i, j))
                nonEdgePolyData = self.deletePoint(nonEdgePolyData, edgeIds)

            rippedBreastPolyData = nonEdgePolyData
            self.createNewModelNode(rippedBreastPolyData, "Ripped_BreastPolyData_{}".format(i))

            for j in range(5): #邊緣平滑次數
                _, edgeIds = self.extractBoundaryPoints(rippedBreastPolyData, "Edge_{}_Smooth_{}".format(i, j))
                rippedBreastPolyData = self.smoothBoundary(rippedBreastPolyData, edgeIds)

            #取得平滑後的邊緣
            edgePolydata, _ = self.extractBoundaryPoints(rippedBreastPolyData, "Edge_Final_{}".format(i))
            boundaryMesh = self.createBoundaryMesh(edgePolydata)
            self.createNewModelNode(boundaryMesh, "BoundaryMesh_{}".format(i))
            smoothedBoundaryMesh = self.smoothBoundaryMesh(boundaryMesh)
            self.createNewModelNode(smoothedBoundaryMesh, "BoundaryMesh_Smoothed_{}".format(i))

            self.createNewModelNode(self.mergeBreastAndBoundary(rippedBreastPolyData, smoothedBoundaryMesh), "MergedPolyData")

    def refineBreastPolyData(self, polyData, holeSize):
        holeFiller = vtk.vtkFillHolesFilter()
        holeFiller.SetInputData(polyData)
        holeFiller.SetHoleSize(holeSize)
        holeFiller.Update()

        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputConnection(holeFiller.GetOutputPort())
        normalFilter.ComputePointNormalsOn()
        normalFilter.SplittingOff()
        normalFilter.Update()

        return normalFilter.GetOutput()

    def extractBoundaryPoints(self, polyData, edgeName = ""):
        idFilter = vtk.vtkIdFilter()
        idFilter.SetInputData(polyData)
        idFilter.SetIdsArrayName("ids")
        idFilter.PointIdsOn()
        idFilter.CellIdsOff()
        idFilter.Update()
        
        edgeFilter = vtk.vtkFeatureEdges()
        edgeFilter.SetInputConnection(idFilter.GetOutputPort())
        edgeFilter.BoundaryEdgesOn()
        edgeFilter.FeatureEdgesOff()
        edgeFilter.ManifoldEdgesOff()
        edgeFilter.NonManifoldEdgesOff()
        edgeFilter.Update()

        if edgeName != "":
            #print("Name:{} - Edge Id List:".format(edgeName))
            #print(vtk_to_numpy(edgeFilter.GetOutput().GetPointData().GetArray("ids")))
            self.createNewModelNode(edgeFilter.GetOutput(), edgeName)

        return edgeFilter.GetOutput(), vtk_to_numpy(edgeFilter.GetOutput().GetPointData().GetArray("ids"))
    
    def smoothBoundary(self, polyData, edgeIds):
        nonEdgePolyData = self.deletePoint(polyData, edgeIds)

        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(polyData)
        smoothFilter.SetNumberOfIterations(150)
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.SetEdgeAngle(180)
        smoothFilter.SetRelaxationFactor(0.5)
        smoothFilter.SetSourceData(nonEdgePolyData)
        smoothFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(smoothFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        return cleanFilter.GetOutput()
    
    def createBoundaryMesh(self, polyData):
        contourFilter = vtk.vtkContourTriangulator()
        contourFilter.SetInputData(polyData)
        contourFilter.Update()

        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.SetInputConnection(contourFilter.GetOutputPort())
        cleanPolyData.SetTolerance(0.0001)
        cleanPolyData.Update()

        subDivisionFilter = vtk.vtkLoopSubdivisionFilter()
        subDivisionFilter.SetNumberOfSubdivisions(3)
        subDivisionFilter.SetInputConnection(cleanPolyData.GetOutputPort())
        subDivisionFilter.Update()

        return subDivisionFilter.GetOutput()
    
    def smoothBoundaryMesh(self, polyData):
        edgePolyData, _ = self.extractBoundaryPoints(polyData)

        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(polyData)
        smoothFilter.SetNumberOfIterations(50)
        smoothFilter.SetRelaxationFactor(0.005)
        smoothFilter.SetSourceData(edgePolyData)
        smoothFilter.Update()

        return smoothFilter.GetOutput()
    
    def mergeBreastAndBoundary(self, breastPolyData, boundaryPolyData):
        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputData(boundaryPolyData)
        normalFilter.FlipNormalsOn()
        normalFilter.Update()

        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(breastPolyData)
        appendFilter.AddInputData(normalFilter.GetOutput())
        appendFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        holeFilter = vtk.vtkFillHolesFilter()
        holeFilter.SetInputConnection(cleanFilter.GetOutputPort())
        holeFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(holeFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputConnection(cleanFilter.GetOutputPort())
        normalFilter.Update()

        return normalFilter.GetOutput()

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
        #self.assertIsNotNone(logic.hasImageData(volumeNode))
        self.delayDisplay('Test passed!')
