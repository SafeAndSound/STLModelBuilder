import logging
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
from vtk.util import numpy_support

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

        # Default Parameters
        self.buildOBJLayout()
        self.buildSTLLayout()

        self.OBJLogic = OBJModelBuilderLogic()
        self.OBJLogic.initiate(self)

        self.STLLogic = STLModelBuilderLogic()
        self.STLLogic.initiate(self)
    
    def buildOBJLayout(self):
        # Default Parameters
        self.targetColor = qt.QColor("#4573a0")

        # Parameters Area
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "OBJ Process Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # input volume selector
        self.OBJModelSelector = slicer.qMRMLNodeComboBox()
        self.OBJModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.OBJModelSelector.selectNodeUponCreation = True
        self.OBJModelSelector.addEnabled = False
        self.OBJModelSelector.removeEnabled = False
        self.OBJModelSelector.noneEnabled = False
        self.OBJModelSelector.showHidden = False
        self.OBJModelSelector.showChildNodeTypes = False
        self.OBJModelSelector.setMRMLScene(slicer.mrmlScene)
        self.OBJModelSelector.setToolTip("Model node containing geometry and texture coordinates.")
        parametersFormLayout.addRow("Input OBJ Model: ", self.OBJModelSelector)

        # input texture selector
        self.OBJTextureSelector = slicer.qMRMLNodeComboBox()
        self.OBJTextureSelector.nodeTypes = ["vtkMRMLVectorVolumeNode"]
        self.OBJTextureSelector.addEnabled = False
        self.OBJTextureSelector.removeEnabled = False
        self.OBJTextureSelector.noneEnabled = False
        self.OBJTextureSelector.showHidden = False
        self.OBJTextureSelector.showChildNodeTypes = False
        self.OBJTextureSelector.setMRMLScene(slicer.mrmlScene)
        self.OBJTextureSelector.setToolTip("Color image containing texture image.")
        parametersFormLayout.addRow("Texture: ", self.OBJTextureSelector)

        # inpute color selector
        self.colorButton = qt.QPushButton()
        self.colorButton.setStyleSheet("background-color: " + self.targetColor.name())
        parametersFormLayout.addRow("Marker Color:", self.colorButton)

        # Texture Button
        self.textureButton = qt.QPushButton("Apply Texture")
        self.textureButton.toolTip = "Paste the texture onto the model."
        self.textureButton.enabled = False
        parametersFormLayout.addRow(self.textureButton)

        # Apply Button
        self.preprocessButton = qt.QPushButton("Remove Tape")
        self.preprocessButton.toolTip = "Remove tapes from the model."
        self.preprocessButton.enabled = False
        parametersFormLayout.addRow(self.preprocessButton)

        # Select Breasts
        self.breastButton = qt.QPushButton("Finish Select Breasts")
        self.breastButton.toolTip = "Click after breasts are selected."
        self.breastButton.enabled = False
        parametersFormLayout.addRow(self.breastButton)

        # Cleanup Breasts
        self.OBJCleanupButton = qt.QPushButton("Cleanup Results and Restart")
        self.OBJCleanupButton.toolTip = "Remove every created models"
        self.OBJCleanupButton.enabled = True
        parametersFormLayout.addRow(" ", None)
        parametersFormLayout.addRow(" ", None)
        parametersFormLayout.addRow(self.OBJCleanupButton)

        # connections
        self.OBJModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelectOBJInputData)
        self.OBJTextureSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelectOBJInputData)
        self.colorButton.connect("clicked(bool)", self.onSelectColor)
        self.textureButton.connect("clicked(bool)", self.onTextureButton)
        self.preprocessButton.connect("clicked(bool)", self.onPreprocessButton)
        self.breastButton.connect("clicked(bool)", self.onBreastButton)
        self.OBJCleanupButton.connect("clicked(bool)", self.onCleanupOBJButton)

        self.onSelectOBJInputData()
    
    def buildSTLLayout(self):
        # Default Parameters

        # Parameters Area
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "STL Process Parameters"
        self.layout.addWidget(parametersCollapsibleButton)
        self.layout.addStretch(1)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # input volume selector
        self.STLLeftModelSelector = slicer.qMRMLNodeComboBox()
        self.STLLeftModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.STLLeftModelSelector.selectNodeUponCreation = True
        self.STLLeftModelSelector.addEnabled = False
        self.STLLeftModelSelector.removeEnabled = False
        self.STLLeftModelSelector.noneEnabled = False
        self.STLLeftModelSelector.showHidden = False
        self.STLLeftModelSelector.showChildNodeTypes = False
        self.STLLeftModelSelector.setMRMLScene(slicer.mrmlScene)
        self.STLLeftModelSelector.setToolTip("Model node containing geometry and texture coordinates.")
        parametersFormLayout.addRow("Input Left STL Model: ", self.STLLeftModelSelector)

        # input volume selector
        self.STLRightModelSelector = slicer.qMRMLNodeComboBox()
        self.STLRightModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.STLRightModelSelector.selectNodeUponCreation = True
        self.STLRightModelSelector.addEnabled = False
        self.STLRightModelSelector.removeEnabled = False
        self.STLRightModelSelector.noneEnabled = False
        self.STLRightModelSelector.showHidden = False
        self.STLRightModelSelector.showChildNodeTypes = False
        self.STLRightModelSelector.setMRMLScene(slicer.mrmlScene)
        self.STLRightModelSelector.setToolTip("Model node containing geometry and texture coordinates.")
        parametersFormLayout.addRow("Input Right STL Model: ", self.STLRightModelSelector)

        # Apply Button
        self.STLApplyButton = qt.QPushButton("Start Processing")
        self.STLApplyButton.toolTip = "Run the algorithm."
        self.STLApplyButton.enabled = False
        parametersFormLayout.addRow(self.STLApplyButton)

        # Cleanup Breasts
        self.STLCleanupButton = qt.QPushButton("Cleanup Results and Restart")
        self.STLCleanupButton.toolTip = "Remove every created models"
        self.STLCleanupButton.enabled = True
        parametersFormLayout.addRow(" ", None)
        parametersFormLayout.addRow(" ", None)
        parametersFormLayout.addRow(self.STLCleanupButton)

        # connections
        self.STLLeftModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelectSTLInputData)
        self.STLRightModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelectSTLInputData)
        self.STLApplyButton.connect("clicked(bool)", self.onProcessSeperateBreast)
        self.STLCleanupButton.connect("clicked(bool)", self.onCleanupSTLButton)

        self.onSelectSTLInputData()

    def cleanup(self):
        pass

    # OBJ Process Functions
    def onSelectOBJInputData(self):
        self.textureButton.enabled = self.OBJModelSelector.currentNode() and self.OBJTextureSelector.currentNode()
        self.preprocessButton.enabled = self.OBJModelSelector.currentNode() and self.OBJTextureSelector.currentNode()

    def onSelectColor(self):
        self.targetColor = qt.QColorDialog.getColor()
        self.colorButton.setStyleSheet("background-color: " + self.targetColor.name())
        self.colorButton.update()

    def onTextureButton(self):
        self.OBJLogic.showTextureOnModel(self.OBJModelSelector.currentNode(), self.OBJTextureSelector.currentNode())

    def onPreprocessButton(self):
        self.OBJLogic.startPreprocessing(self.OBJModelSelector.currentNode(), self.OBJTextureSelector.currentNode(), self.targetColor)

    def onBreastButton(self):
        self.OBJLogic.truncateBreastPolyData()

    def finishPreProcessing(self):
        self.breastButton.enabled = True
        self.OBJLogic.setupFiducialNodeOperation()
    
    def onCleanupOBJButton(self):
        self.OBJLogic.cleanup()
    
    # STL Process Functions
    def onSelectSTLInputData(self):
        self.STLApplyButton.enabled = self.STLLeftModelSelector.currentNode() and self.STLRightModelSelector.currentNode()
    
    def onProcessSeperateBreast(self):
        self.STLLogic.startProcessing(self.STLLeftModelSelector.currentNode(), self.STLRightModelSelector.currentNode())
    
    def onCleanupSTLButton(self):
        self.STLLogic.cleanup()

    def onReload(self):
        ScriptedLoadableModuleWidget.onReload(self)


class OBJModelBuilderLogic(ScriptedLoadableModuleLogic):
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
        self.createdModelNodes = []
        self.markerName = "Reference_Breast_Position"
        self.mainModelNode = None

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
        textureImageFlipVert.SetInputConnection(
            textureImageNode.GetImageDataConnection())
        textureImageFlipVert.Update()
        textureImageData = textureImageFlipVert.GetOutput()
        pointData = polyData.GetPointData()
        tcoords = pointData.GetTCoords()
        numOfPoints = pointData.GetNumberOfTuples()
        assert numOfPoints == tcoords.GetNumberOfTuples(
        ), "Number of texture coordinates does not equal number of points"
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
            colorArray.SetTuple3(
                pointIndex, rgb[0]/255., rgb[1]/255., rgb[2]/255.)
            colorArray.Modified()
            pointData.AddArray(colorArray)

        pointData.Modified()
        polyData.Modified()

    def startPreprocessing(self, modelNode, textureImageNode, targetColor):
        """
        Run the actual algorithm
        """
        print("----Start Processing----")
        startTime = time.time()
        print("Start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(startTime)) + "\n")

        self.mainModelNode = modelNode

        newPolyData = vtk.vtkPolyData()
        newPolyData.DeepCopy(modelNode.GetPolyData())
        self.modifidedModelNode = self.createNewModelNode(newPolyData, "Modified_Model")
        newModelNode = self.modifidedModelNode

        # 轉換顏色格式:QColor -> np.array
        targetColor = np.array([targetColor.redF(), targetColor.greenF(), targetColor.blueF()])
        print("Selected Color: {}".format(targetColor))

        # 取得vtkMRMLModelNode讀取的檔案
        fileName = modelNode.GetStorageNode().GetFileName()
        print("OBJ File Path: {}\n".format(fileName))

        print("Origin Model points: {}".format(
            self.modifidedModelNode.GetPolyData().GetNumberOfPoints()))
        print("Origin Model cells: {}\n".format(
            self.modifidedModelNode.GetPolyData().GetNumberOfCells()))

        # 產生點的顏色資料
        self.convertTextureToPointAttribute(newModelNode, textureImageNode)

        # 取出顏色於範圍內的點id
        delPointIds = self.extractSelection(newModelNode, targetColor, 0.13)

        # 刪除顏色符合的點
        newModelNode.SetAndObservePolyData(
            self.deletePoint(newModelNode.GetPolyData(), delPointIds))

        # 處理PolyData (降低面數、破洞處理......)
        self.reduceAndCleanPolyData(newModelNode)

        print("Modified Model points: {}".format(
            newModelNode.GetPolyData().GetNumberOfPoints()))
        print("Modified Model cells: {}\n".format(
            newModelNode.GetPolyData().GetNumberOfCells()))

        modelNode.GetDisplayNode().VisibilityOff()
        newModelNode.GetDisplayNode().VisibilityOn()

        self.widget.finishPreProcessing()

        print("\n----Complete Processing----")
        stopTime = time.time()
        print("Complete time: " +
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stopTime)))
        logging.info('Processing completed in {0:.2f} seconds\n'.format(
            stopTime - startTime))

        return True

    def reduceAndCleanPolyData(self, modelNode):
        # triangulate
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(modelNode.GetPolyData())
        triangleFilter.Update()

        # decimate
        decimateFilter = vtk.vtkDecimatePro()
        decimateFilter.SetInputConnection(triangleFilter.GetOutputPort())
        decimateFilter.SetTargetReduction(0.33)
        decimateFilter.PreserveTopologyOn()
        decimateFilter.BoundaryVertexDeletionOff()
        decimateFilter.Update()

        # clean
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(decimateFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        # relax
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

        # normal
        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputConnection(relaxFilter.GetOutputPort())
        normalFilter.ComputePointNormalsOn()
        normalFilter.SplittingOff()
        normalFilter.Update()

        # alignCenter
        polyData = normalFilter.GetOutput()
        points_array = vtk_to_numpy(polyData.GetPoints().GetData())
        center = points_array.sum(axis=0) / points_array.shape[0]
        np.copyto(points_array, points_array - center)
        polyData.GetPoints().GetData().Modified()

        modelNode.SetAndObservePolyData(polyData)

    def extractSelection(self, modelNode, targetColor, threshold):
        colorData = vtk_to_numpy(
            modelNode.GetPolyData().GetPointData().GetArray("Color"))
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

    def setupFiducialNodeOperation(self):
        # Create fiducial node
        fiducialNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLMarkupsFiducialNode())
        fiducialNode.SetName(self.markerName)

        placeModePersistence = 1
        slicer.modules.markups.logic().StartPlaceMode(placeModePersistence)

    def truncateBreastPolyData(self):
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SwitchToViewTransformMode()
        # also turn off place mode persistence if required
        interactionNode.SetPlaceModePersistence(0)

        fiducialNode = slicer.util.getNode(self.markerName)
        numFids = fiducialNode.GetNumberOfFiducials()

        breastPos = []
        # Use the last 2 fiducials as two breast positions
        for i in range(numFids - 2, numFids):
            ras = [0, 0, 0]
            fiducialNode.GetNthFiducialPosition(i, ras)
            breastPos.append(ras)
            # the world position is the RAS position with any transform matrices applied
            world = [0, 0, 0, 0]
            fiducialNode.GetNthFiducialWorldCoordinates(i, world)
            #print(i, ": RAS =", ras, ", world =", world)

        slicer.mrmlScene.RemoveNode(fiducialNode)

        wallGenerator = BreastWallGenerator()

        for i in range(2):
            # 尋找最接近選取點的mesh
            connectFilter = vtk.vtkPolyDataConnectivityFilter()
            connectFilter.SetInputData(self.modifidedModelNode.GetPolyData())
            connectFilter.SetExtractionModeToClosestPointRegion()
            connectFilter.SetClosestPoint(breastPos[i])
            connectFilter.Update()

            rawBreastPolyData = connectFilter.GetOutput()
            self.createNewModelNode(connectFilter.GetOutput(), "Breast_{}".format(i))

            mergedPolyData = wallGenerator.generateWall(rawBreastPolyData, True)

            self.createNewModelNode(mergedPolyData, "MergedPolyData")

    """
    def calculateArea(self, polyData):
        Point_cordinates = polyData.GetPoints().GetData()
        numpy_coordinates = numpy_support.vtk_to_numpy(Point_cordinates)
        print(numpy_coordinates.shape)

        vertexes = [polyData.GetPoint(i)
                    for i in range(polyData.GetNumberOfPoints())]
        pdata = polyData.GetPolys().GetData()
        values = [int(pdata.GetTuple1(i))
                  for i in range(pdata.GetNumberOfTuples())]
        triangles = []
        while values:
            n = values[0]  # number of points in the polygon
            triangles.append(values[1:n + 1])
            del values[0:n + 1]

        area = 0.0
        for j in range(20):
            k = random.randint(0, len(triangles))
            # 點a
            ax = numpy_coordinates[triangles[k][0]][0]
            ay = numpy_coordinates[triangles[k][0]][1]
            az = numpy_coordinates[triangles[k][0]][2]
            # 點b
            bx = numpy_coordinates[triangles[k][1]][0]
            by = numpy_coordinates[triangles[k][1]][1]
            bz = numpy_coordinates[triangles[k][1]][2]
            # 點c
            cx = numpy_coordinates[triangles[k][2]][0]
            cy = numpy_coordinates[triangles[k][2]][1]
            cz = numpy_coordinates[triangles[k][2]][2]
            x = np.sqrt(np.square(bx - ax) +
                        np.square(by - ay) + np.square(bz - az))
            y = np.sqrt(np.square(cx - ax) +
                        np.square(cy - ay) + np.square(cz - az))
            z = np.sqrt(np.square(bx - cx) +
                        np.square(by - cy) + np.square(bz - cz))
            s = float(x + y + z) / 2
            tmp = np.sqrt(s * (s - x) * (s - y) * (s - z))
            #print("tmp=", tmp)
            area += tmp
        area = float(area)/20

        return area
    """

    def createNewModelNode(self, polyData, nodeName):
        modelNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLModelNode())
        modelNode.SetName(nodeName)
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(polyData)

        self.createdModelNodes.append(modelNode)

        return modelNode

    def cleanup(self):
        for node in self.createdModelNodes:
            slicer.mrmlScene.RemoveNode(node)

        while True:
            try:
                slicer.util.getNode(self.markerName)
                slicer.mrmlScene.RemoveNode(slicer.util.getNode(self.markerName))
            except:
                break
        
        if self.mainModelNode:
            self.mainModelNode.GetDisplayNode().VisibilityOn()

class STLModelBuilderLogic(ScriptedLoadableModuleLogic):
    def initiate(self, widget):
        self.widget = widget
        self.createdModelNodes = []
        self.mainLeftBreastModelNode = None
        self.mainRightBreastModelNode = None

    def startProcessing(self, leftBreastModelNode, rightBreastModelNode):
        """
        Run the actual algorithm
        """
        print("----Start Processing----")
        startTime = time.time()
        print("Start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(startTime)) + "\n")

        self.mainLeftBreastModelNode = leftBreastModelNode
        self.mainRightBreastModelNode = rightBreastModelNode

        wallGenerator = BreastWallGenerator()

        #檢查是否只有單邊乳房
        hasLeftBreast = True
        hasRightBreast = True
        if leftBreastModelNode.GetName() == rightBreastModelNode.GetName():
            name = leftBreastModelNode.GetName().lower()
            hasLeftBreast = ('left' in name)
            hasRightBreast = ('right' in name)
            if not hasLeftBreast and not hasRightBreast:
                hasLeftBreast = True

        if hasLeftBreast:
            leftBreastPolyData = vtk.vtkPolyData()
            leftBreastPolyData.DeepCopy(leftBreastModelNode.GetPolyData())
            fileName = leftBreastModelNode.GetStorageNode().GetFileName()
            print("Left Breast File Path: {}".format(fileName))
            print("Origin Left Breast Points: {}".format(leftBreastPolyData.GetNumberOfPoints()))
            leftBreastPolyData = self.reduceAndCleanPolyData(leftBreastPolyData)
            print("Modified Left Breast Points: {}".format(leftBreastPolyData.GetNumberOfPoints()))
            leftBreastModelNode.GetDisplayNode().VisibilityOff()
            leftBreastPolyData = wallGenerator.generateWall(leftBreastPolyData, False)
            self.createNewModelNode(leftBreastPolyData, "MergedLeftPolyData")

        if hasRightBreast:
            rightBreastPolyData = vtk.vtkPolyData()
            rightBreastPolyData.DeepCopy(rightBreastModelNode.GetPolyData())
            fileName = rightBreastModelNode.GetStorageNode().GetFileName()
            print("Right Breast File Path: {}".format(fileName))
            print("Origin Right Breast Points: {}".format(rightBreastPolyData.GetNumberOfPoints()))
            rightBreastPolyData = self.reduceAndCleanPolyData(rightBreastPolyData)
            print("Modified Right Breast Points:: {}".format(rightBreastPolyData.GetNumberOfPoints()))
            rightBreastModelNode.GetDisplayNode().VisibilityOff()
            rightBreastPolyData = wallGenerator.generateWall(rightBreastPolyData, False)
            self.createNewModelNode(rightBreastPolyData, "MergedRightPolyData")

        print("\n----Complete Processing----")
        stopTime = time.time()
        print("Complete time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stopTime)))
        logging.info('Processing completed in {0:.2f} seconds\n'.format(stopTime - startTime))

        return True

    def reduceAndCleanPolyData(self, polyData):
        # triangulate
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(polyData)
        triangleFilter.Update()

        # decimate
        decimateFilter = vtk.vtkDecimatePro()
        decimateFilter.SetInputConnection(triangleFilter.GetOutputPort())
        decimateFilter.SetTargetReduction(0.33)
        decimateFilter.PreserveTopologyOn()
        decimateFilter.BoundaryVertexDeletionOff()
        decimateFilter.Update()

        # clean
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(decimateFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        # relax
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

        # normal
        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputConnection(relaxFilter.GetOutputPort())
        normalFilter.ComputePointNormalsOn()
        normalFilter.SplittingOff()
        normalFilter.Update()

        return normalFilter.GetOutput()

    def createNewModelNode(self, polyData, nodeName):
        modelNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLModelNode())
        modelNode.SetName(nodeName)
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(polyData)

        self.createdModelNodes.append(modelNode)

        return modelNode

    def cleanup(self):
        for node in self.createdModelNodes:
            slicer.mrmlScene.RemoveNode(node)

        if self.mainLeftBreastModelNode:
            self.mainLeftBreastModelNode.GetDisplayNode().VisibilityOn()
        if self.mainRightBreastModelNode:
            self.mainRightBreastModelNode.GetDisplayNode().VisibilityOn()

class BreastWallGenerator():
    def generateWall(self, breastPolyData, ripEdge):
        refinedBreastPolyData = self.refineBreastPolyData(breastPolyData, 50)

        if ripEdge:
            rippedBreastPolyData = refinedBreastPolyData
            for _ in range(3):  # 藉由直接移除n層boundary減少突出邊緣
                _, edgeIds = self.extractBoundaryPoints(rippedBreastPolyData)
                rippedBreastPolyData = self.deletePoint(rippedBreastPolyData, edgeIds)

            smoothedBreastPolyData = self.smoothBoundary(rippedBreastPolyData, 2)
        else:
            smoothedBreastPolyData = refinedBreastPolyData

        # 取得平滑後的邊緣
        edgePolydata, _ = self.extractBoundaryPoints(smoothedBreastPolyData)

        wallMesh = self.createWallMesh(edgePolydata)

        mergedPolyData = self.mergeBreastAndBoundary(smoothedBreastPolyData, wallMesh)
        
        return mergedPolyData

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

    def smoothBoundary(self, polyData, edgeWidth):
        nonEdgePolyData = polyData
        for _ in range(edgeWidth):  # 邊緣平滑次數
            _, edgeIds = self.extractBoundaryPoints(nonEdgePolyData)
            nonEdgePolyData = self.deletePoint(nonEdgePolyData, edgeIds)

        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(polyData)
        smoothFilter.SetNumberOfIterations(50)
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.SetEdgeAngle(180)
        smoothFilter.SetRelaxationFactor(1)
        smoothFilter.SetSourceData(nonEdgePolyData)
        smoothFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(smoothFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        return cleanFilter.GetOutput()

    def extractBoundaryPoints(self, polyData, edgeName=""):
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
            self.createNewModelNode(edgeFilter.GetOutput(), edgeName)

        return edgeFilter.GetOutput(), vtk_to_numpy(edgeFilter.GetOutput().GetPointData().GetArray("ids"))
    
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

    def createWallMesh(self, edgePolyData):
        numpy_coordinates = numpy_support.vtk_to_numpy(edgePolyData.GetPoints().GetData())
        print('size=', numpy_coordinates.shape, numpy_coordinates.size)
        # print(numpy_coordinates)
        originPointCount = int(numpy_coordinates.shape[0])
        
        edgePolyData.GetLines().InitTraversal()
        pointIdLink = [-1] * edgePolyData.GetNumberOfLines()
        idList = vtk.vtkIdList()
        while edgePolyData.GetLines().GetNextCell(idList):
            if pointIdLink[idList.GetId(0)] != -1:
                print("line from id {} already assign!!".format(idList.GetId(0)))
            else:
                pointIdLink[idList.GetId(0)] = idList.GetId(1)

        pointIdList = [0]
        while pointIdLink[pointIdList[-1]] != 0 and pointIdLink[pointIdList[-1]] != 1:
            next = pointIdLink[pointIdList[-1]]
            pointIdList.append(next)
        
        print(len(pointIdList))

        t = 1000
        polyPoints = vtk.vtkPoints()
        polyPoints.DeepCopy(edgePolyData.GetPoints())

        points = list(range(originPointCount))
        appear = []
        for i in range(t):
            while True:
                random.shuffle(points)
                avgp = (numpy_coordinates[points[0]] + numpy_coordinates[points[1]] + numpy_coordinates[points[2]]) / 3
                h = hash(str(avgp))
                if h not in appear:
                    polyPoints.InsertPoint(originPointCount + i, avgp)
                    appear.append(h)
                    break

        originData = vtk.vtkPolyData()
        originData.SetPoints(polyPoints)

        constrain = vtk.vtkPolyData()
        constrain.SetPoints(polyPoints)
        constrain.SetPolys(vtk.vtkCellArray())

        delaunayFilter = vtk.vtkDelaunay2D()
        delaunayFilter.SetInputData(originData)
        delaunayFilter.SetSourceData(constrain)
        delaunayFilter.SetTolerance(0.01)
        delaunayFilter.SetProjectionPlaneMode(vtk.VTK_BEST_FITTING_PLANE)
        delaunayFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(delaunayFilter.GetOutputPort())
        cleanFilter.Update()

        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputConnection(cleanFilter.GetOutputPort())
        smoothFilter.SetNumberOfIterations(200)
        smoothFilter.SetRelaxationFactor(0.05)
        smoothFilter.BoundarySmoothingOff()
        smoothFilter.Update()

        subdivisionFilter = vtk.vtkLoopSubdivisionFilter()
        subdivisionFilter.SetNumberOfSubdivisions(2)
        subdivisionFilter.SetInputConnection(smoothFilter.GetOutputPort())
        subdivisionFilter.Update()

        return subdivisionFilter.GetOutput()

    def mergeBreastAndBoundary(self, breastPolyData, wallPolyData):
        # 先移除最外圍的點 避免與胸部data重疊
        _, edgeIds = self.extractBoundaryPoints(wallPolyData)
        rippedWallPolyData = self.deletePoint(wallPolyData, edgeIds)
        rippedWallEdge, _ = self.extractBoundaryPoints(rippedWallPolyData)
        wallStrips = vtk.vtkStripper()
        wallStrips.SetInputData(rippedWallEdge)
        wallStrips.Update()
        edge1 = wallStrips.GetOutput()

        breastEdge, _ = self.extractBoundaryPoints(breastPolyData)
        boundaryStrips = vtk.vtkStripper()
        boundaryStrips.SetInputData(breastEdge)
        boundaryStrips.Update()
        edge2 = boundaryStrips.GetOutput()

        stitcer = PolyDataStitcher()
        stitchPolyData = stitcer.stitch(edge1, edge2)
        #self.createNewModelNode(stitchPolyData, "Stitch")

        #先將胸壁與縫合面合併
        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(rippedWallPolyData)
        appendFilter.AddInputData(stitchPolyData)
        appendFilter.Update()

        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputConnection(appendFilter.GetOutputPort())
        normalFilter.Update()
        
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(normalFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        holeFilter = vtk.vtkFillHolesFilter()
        holeFilter.SetInputConnection(cleanFilter.GetOutputPort())
        holeFilter.Update()

        #self.createNewModelNode(holeFilter.GetOutput(), "Stitch_Combine")

        #再次合併胸壁和胸部
        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(breastPolyData)
        appendFilter.AddInputData(holeFilter.GetOutput())
        appendFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        connectFilter = vtk.vtkPolyDataConnectivityFilter()
        connectFilter.SetInputConnection(cleanFilter.GetOutputPort())
        connectFilter.SetExtractionModeToLargestRegion()
        connectFilter.Update()

        holeFilter = vtk.vtkFillHolesFilter()
        holeFilter.SetInputConnection(connectFilter.GetOutputPort())
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

    def createNewModelNode(self, polyData, nodeName):
        modelNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLModelNode())
        modelNode.SetName(nodeName)
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(polyData)

        return modelNode

class PolyDataStitcher():
    def extract_points(self, source):
        # Travers the cells and add points while keeping their order.
        points = source.GetPoints()
        cells = source.GetLines()
        cells.InitTraversal()
        idList = vtk.vtkIdList()
        pointIds = []
        while cells.GetNextCell(idList):
            for i in range(0, idList.GetNumberOfIds()):
                pId = idList.GetId(i)
                # Only add the point id if the previously added point does not
                # have the same id. Avoid p->p duplications which occur for example
                # if a poly-line is traversed. However, other types of point
                # duplication currently are not avoided: a->b->c->a->d
                if len(pointIds) == 0 or pointIds[-1] != pId:
                    pointIds.append(pId)
        result = []
        for i in pointIds:
            result.append(points.GetPoint(i))
        return result

    def reverse_lines(self, source):
        strip = vtk.vtkStripper()
        strip.SetInputData(source)
        strip.Update()
        reversed = vtk.vtkReverseSense()
        reversed.SetInputConnection(strip.GetOutputPort())
        reversed.Update()
        return reversed.GetOutput()

    def find_closest_point(self, points, samplePoint):
        points = np.asarray(points)
        assert(len(points.shape) == 2 and points.shape[1] == 3)
        nPoints = points.shape[0]
        diff = np.array(points) - np.tile(samplePoint, [nPoints, 1])
        pId = np.argmin(np.linalg.norm(diff, axis=1))
        return pId

    def stitch(self, edge1, edge2):
        # Extract points along the edge line (in correct order).
        # The following further assumes that the polyline has the
        # same orientation (clockwise or counterclockwise).
        edge2 = self.reverse_lines(edge2)

        points1 = self.extract_points(edge1)
        points2 = self.extract_points(edge2)
        n1 = len(points1)
        n2 = len(points2)

        # Prepare result containers.
        # Variable points concatenates points1 and points2.
        # Note: all indices refer to this targert container!
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        points.SetNumberOfPoints(n1+n2)
        for i, p1 in enumerate(points1):
            points.SetPoint(i, p1)
        for i, p2 in enumerate(points2):
            points.SetPoint(i+n1, p2)

        # The following code stitches the curves edge1 with (points1) and
        # edge2 (with points2) together based on a simple growing scheme.

        # Pick a first stitch between points1[0] and its closest neighbor
        # of points2.
        i1Start = 0
        i2Start = self.find_closest_point(points2, points1[i1Start])
        i2Start += n1  # offset to reach the points2

        # Initialize
        i1 = i1Start
        i2 = i2Start
        p1 = np.asarray(points.GetPoint(i1))
        p2 = np.asarray(points.GetPoint(i2))
        mask = np.zeros(n1+n2, dtype=bool)
        count = 0
        while not np.all(mask):
            count += 1
            i1Candidate = (i1+1) % n1
            i2Candidate = (i2+1-n1) % n2+n1
            p1Candidate = np.asarray(points.GetPoint(i1Candidate))
            p2Candidate = np.asarray(points.GetPoint(i2Candidate))
            diffEdge12C = np.linalg.norm(p1-p2Candidate)
            diffEdge21C = np.linalg.norm(p2-p1Candidate)

            mask[i1] = True
            mask[i2] = True
            if diffEdge12C < diffEdge21C:
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, i1)
                triangle.GetPointIds().SetId(1, i2)
                triangle.GetPointIds().SetId(2, i2Candidate)
                cells.InsertNextCell(triangle)
                i2 = i2Candidate
                p2 = p2Candidate
            else:
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, i1)
                triangle.GetPointIds().SetId(1, i2)
                triangle.GetPointIds().SetId(2, i1Candidate)
                cells.InsertNextCell(triangle)
                i1 = i1Candidate
                p1 = p1Candidate

        # Add the last triangle.
        i1Candidate = (i1+1) % n1
        i2Candidate = (i2+1-n1) % n2+n1
        if (i1Candidate <= i1Start) or (i2Candidate <= i2Start):
            if i1Candidate <= i1Start:
                iC = i1Candidate
            else:
                iC = i2Candidate
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, i1)
            triangle.GetPointIds().SetId(1, i2)
            triangle.GetPointIds().SetId(2, iC)
            cells.InsertNextCell(triangle)

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)
        poly.BuildLinks()

        return poly

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
        # self.assertIsNotNone(logic.hasImageData(volumeNode))
        self.delayDisplay('Test passed!')
