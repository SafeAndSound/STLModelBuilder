import sys
import os
import unittest
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
from vtk.util.numpy_support import vtk_to_numpy
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

        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Start Processing")
        self.applyButton.toolTip = "Run the algorithm."
        self.applyButton.enabled = False
        parametersFormLayout.addRow(self.applyButton)

        # connections
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.inputModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.inputTextureSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

        # Add vertical spacer
        self.layout.addStretch(1)

        # Refresh Apply button state
        self.onSelect()

    def cleanup(self):
        pass

    def onSelect(self):
        self.applyButton.enabled = self.inputTextureSelector.currentNode() and self.inputModelSelector.currentNode()

    def onApplyButton(self):
        logic = STLModelBuilderLogic()
        logic.run(self.inputModelSelector.currentNode(), self.inputTextureSelector.currentNode())

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

    def hasImageData(self, volumeNode):
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

    def run(self, modelNode, textureImageNode):
        """
        Run the actual algorithm
        """
        print("----Start Processing----")
        startTime = time.time()
        print("Start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(startTime)) + "\n")

        #取得vtkMRMLModelNode讀取的檔案
        fileName = modelNode.GetStorageNode().GetFileName()
        print("OBJ File Path: {}".format(fileName))

        #產生點的顏色資料
        self.convertTextureToPointAttribute(modelNode, textureImageNode)

        #取出顏色資料(可由上一步簡化)
        colorData = modelNode.GetPolyData().GetPointData().GetArray("Color")
        colorData_np = vtk_to_numpy(colorData)
        print(colorData_np)
        print(colorData_np[25671])

        #取出顏色於範圍內的點id
        delPointIds = self.extractSelection(modelNode, colorData_np, np.array([0.30588235, 0.4745098,  0.64313725]), 0.5)

        #刪除顏色符合的點
        self.deletePoint(modelNode, delPointIds)

        self.preprocessPolyData(modelNode)

        self.showTextureOnModel(modelNode, textureImageNode)

        print("\n----Complete Processing----")
        stopTime = time.time()
        print("Complete time: " +
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stopTime)))
        logging.info('Processing completed in {0:.2f} seconds'.format(stopTime - startTime))

        return True
    
    def preprocessPolyData(self, modelNode):
        #decimate
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(modelNode.GetPolyData())
        triangleFilter.Update()
        
        decimateFilter = vtk.vtkDecimatePro()
        decimateFilter.SetInputConnection(triangleFilter.GetOutputPort())
        decimateFilter.SetTargetReduction(0.25)
        decimateFilter.PreserveTopologyOn()
        decimateFilter.BoundaryVertexDeletionOff()
        decimateFilter.Update()

        #clean
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(decimateFilter.GetOutputPort())
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

        #connect
        connectFilter = vtk.vtkPolyDataConnectivityFilter()
        connectFilter.SetInputConnection(relaxFilter.GetOutputPort())
        connectFilter.SetExtractionModeToLargestRegion()
        connectFilter.Update()

        #normal
        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputConnection(connectFilter.GetOutputPort())
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

    def extractSelection(self, modelNode, colorData, targetColor, threshold):
        targetId = np.asarray(np.where(np.linalg.norm(colorData - targetColor, axis=1, keepdims=True) < threshold))[0]
        print(targetId)
        
        return targetId

    def deletePoint(self, modelNode, delPointIds):
        #會破壞texcoord 有改善空間
        polyData = modelNode.GetPolyData()
        pPoints = vtk.vtkPoints()
        cellArray = vtk.vtkCellArray()

        oldPoints = vtk_to_numpy(polyData.GetPoints().GetData())
        oldNumberOfPoints = oldPoints.shape[0]
        print(oldNumberOfPoints)

        numberOfdelPoints = delPointIds.shape[0]

        #將所有的poly轉為numpy array
        #直接將所有包含delPointIds的cell移除
        cells = polyData.GetPolys()
        numberOfCells = cells.GetNumberOfCells()
        array = cells.GetData()
        assert(array.GetNumberOfValues() % numberOfCells == 0)
        pointPerCell = array.GetNumberOfValues() // numberOfCells #1 + n
        numpy_cells = vtk_to_numpy(array)
        numpy_cells = np.delete(numpy_cells.reshape((-1, pointPerCell)), 0, axis=1)
        pointPerCell -= 1

        cumulate = 0
        idMapping = [0] * oldNumberOfPoints
        for pid in range(oldNumberOfPoints):
            idMapping[pid] = pid - cumulate

            if cumulate == numberOfdelPoints:
                pPoints.InsertNextPoint(oldPoints[pid])
                continue

            #移除在delPointIds中的點
            if pid != delPointIds[cumulate]:
                pPoints.InsertNextPoint(oldPoints[pid])

            #計算移動後的點id
            if pid == delPointIds[cumulate]:
                cumulate += 1

        for cell in numpy_cells:
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(pointPerCell)
            discard = False

            for pid in range(pointPerCell):
                if cell[pid] in delPointIds: #cell shoud be discard
                    discard = True
                    break

                polygon.GetPointIds().SetId(pid, idMapping[cell[pid]])
    
            if not discard:
                cellArray.InsertNextCell(polygon)

        polyData.SetPoints(pPoints)
        polyData.SetPolys(cellArray)
        polyData.Modified()

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
        self.assertIsNotNone(logic.hasImageData(volumeNode))
        self.delayDisplay('Test passed!')
