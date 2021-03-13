import sys
import os
import unittest
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import logging
import time
import sitkUtils
import SimpleITK as sitk
import numpy as np
import math
import random
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
        self.targetColor = qt.QColor("DarkGray")
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
        self.deletePoint(newModelNode, delPointIds)

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

    """
    def deletePoint(self, modelNode, delPointIds):
        #會破壞texcoord 有改善空間
        polyData = modelNode.GetPolyData()
        points = vtk.vtkPoints()
        cellArray = vtk.vtkCellArray()

        oldPoints = vtk_to_numpy(polyData.GetPoints().GetData())
        oldNumberOfPoints = oldPoints.shape[0]

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
                points.InsertNextPoint(oldPoints[pid])
                continue

            #移除在delPointIds中的點
            if pid != delPointIds[cumulate]:
                points.InsertNextPoint(oldPoints[pid])

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

        polyData.SetPoints(points)
        polyData.SetPolys(cellArray)
        polyData.Modified()
    """

    def deletePoint(self, modelNode, delPointIds):
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(numpy_to_vtk(delPointIds))
        selectionNode.GetProperties().Set(vtk.vtkSelectionNode.INVERSE(), 1)
        selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, modelNode.GetPolyData())
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()

        geometryFilter = vtk.vtkGeometryFilter()
        geometryFilter.SetInputData(extractSelection.GetOutput())
        geometryFilter.Update()

        modelNode.SetAndObservePolyData(geometryFilter.GetOutput())

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
        print("nodename ",nodeName)
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
            print(i, ": RAS =", ras, ", world =", world)
        
        slicer.mrmlScene.RemoveNode(fiducialNode)

        for i in range(2):
            connectFilter = vtk.vtkPolyDataConnectivityFilter()
            connectFilter.SetInputData(self.modifidedModelNode.GetPolyData())
            connectFilter.SetExtractionModeToClosestPointRegion()
            connectFilter.SetClosestPoint(breastPos[i])
            connectFilter.Update()

            newNode = self.createNewModelNode(connectFilter.GetOutput(), "Breast_{}".format(i))

            Point_cordinates = connectFilter.GetOutput().GetPoints().GetData()
            numpy_coordinates = numpy_support.vtk_to_numpy(Point_cordinates)
            print(numpy_coordinates.shape)

            vertexes = [connectFilter.GetOutput().GetPoint(i) for i in range(connectFilter.GetOutput().GetNumberOfPoints())]
            pdata = connectFilter.GetOutput().GetPolys().GetData()
            values = [int(pdata.GetTuple1(i)) for i in range(pdata.GetNumberOfTuples())]
            triangles = []
            while values:
                n = values[0]  # number of points in the polygon
                triangles.append(values[1:n + 1])
                del values[0:n + 1]


            print("ori_area ",self.calculateArea(triangles, numpy_coordinates))

            self.FillPolydataHole(newNode, 50)
            #self.Test(newNode)
            edgePolydata = self.extractBoundaryPoints(newNode, "Edge_{}".format(i))

            #slicer.util.saveNode(newNode, 'C:/Users/sandy/OneDrive/桌面/計畫/present/STLModelBuilder-main/STLModelBuilder/test_{}.vtk'.format(i))
            self.createBoundaryMesh(edgePolydata)

    def Test(self, modelNode): #補洞
        convexHull = vtk.vtkDelaunay3D()
        convexHull.SetInputData(modelNode.GetPolyData())
        outerSurface = vtk.vtkGeometryFilter()
        outerSurface.SetInputConnection(convexHull.GetOutputPort())
        outerSurface.Update()
        modelNode.SetAndObservePolyData(outerSurface.GetOutput())

    def calculateArea(self, triangles, numpy_coordinates):
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
            x = np.sqrt(np.square(bx - ax) + np.square(by - ay) + np.square(bz - az))
            y = np.sqrt(np.square(cx - ax) + np.square(cy - ay) + np.square(cz - az))
            z = np.sqrt(np.square(bx - cx) + np.square(by - cy) + np.square(bz - cz))
            s = float(x + y + z) / 2
            tmp = np.sqrt(s * (s - x) * (s - y) * (s - z))
            print("tmp=", tmp)
            area += tmp
        area = float(area)/20
        return area

    def FillPolydataHole(self, modelNode, holeSize):
        holeFiller = vtk.vtkFillHolesFilter()
        holeFiller.SetInputData(modelNode.GetPolyData())
        holeFiller.SetHoleSize(holeSize)
        holeFiller.Update()
        modelNode.SetAndObservePolyData(holeFiller.GetOutput())

        #modelNode.SetAndObservePolyData(holeFiller.GetOutput())

    def extractBoundaryPoints(self, modelNode, edgeName):
        idFilter = vtk.vtkIdFilter()
        idFilter.SetInputData(modelNode.GetPolyData())
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

        #edgeIds = vtk_to_numpy(edgeFilter.GetOutput().GetPointData().GetArray("ids"))

        self.createNewModelNode(edgeFilter.GetOutput(), edgeName)

        return edgeFilter.GetOutput()
    
    def createBoundaryMesh(self, Polydata):

        Point_cordinates = Polydata.GetPoints().GetData()
        numpy_coordinates = numpy_support.vtk_to_numpy(Point_cordinates)
        print('size=', numpy_coordinates.shape, numpy_coordinates.size)
        # print(numpy_coordinates)

        length = int(numpy_coordinates.shape[0])
        minx = numpy_coordinates[0][0]
        maxx = numpy_coordinates[0][0]
        minz = numpy_coordinates[0][2]
        maxz = numpy_coordinates[0][2]
        miny = numpy_coordinates[0][1]
        maxy = numpy_coordinates[0][1]

        for i in range(1, length):
            if numpy_coordinates[i][0] < minx:
                minx = numpy_coordinates[i][0]
            if numpy_coordinates[i][0] > maxx:
                maxx = numpy_coordinates[i][0]
            if numpy_coordinates[i][1] < miny:
                miny = numpy_coordinates[i][1]
            if numpy_coordinates[i][1] > maxy:
                maxy = numpy_coordinates[i][1]
            if numpy_coordinates[i][2] < minz:
                minz = numpy_coordinates[i][2]
            if numpy_coordinates[i][2] > maxz:
                maxz = numpy_coordinates[i][2]

        avgx = float((minx + maxx) / 2)
        avgy = float((miny + maxy) / 2)
        # avgz = np.median(tmp_list)
        avgz = float((minz + maxz) / 2)
        #print("avg=", avgx, avgy, avgz, minx, maxx, miny, maxy)

        test = Polydata.GetPoints()
        test.InsertPoint(numpy_coordinates.shape[0], avgx, avgy, avgz)

        '''
        ra = float(maxx+minx)/2
        rb = float(maxy+miny)/2
        rc = float(maxz+minz)/2
        print("ra=", ra, ",rb=", rb, ",rc=", rc)
        k = 1
        while k < 101:
            x = random.uniform(-ra, ra)
            y = random.uniform(-rb, rb)
            #x,y座標=(avgx+x,avgy+y)
            z = random.uniform(avgz,maxz)
            z2 = (1-np.square(x)/float(np.square(ra))-np.square(y)/float(np.square(rb)))*np.square(rc)
            z = np.sqrt(z2)+avgz
            test.InsertPoint(numpy_coordinates.shape[0] + k, avgx+x, avgy+y, z)
            k += 1
        '''

        #橢圓
        for i in range(1, 80001):
            k = int(random.uniform(0, length))
            lenx = numpy_coordinates[k][0] - avgx
            leny = numpy_coordinates[k][1] - avgy
            lenz = numpy_coordinates[k][2] - avgz
            rx = random.uniform(0, lenx*0.6)
            ry = random.uniform(0, leny*0.6)
            rz = random.uniform(0, lenz*0.6)
            x = avgx + rx / float(lenx)
            y = avgy + ry / float(leny)
            z = avgz + rz / float(lenz)
            test.InsertPoint(numpy_coordinates.shape[0]+i, x, y, z)
            #for j in range(1, 21):
            #    test.InsertPoint(numpy_coordinates.shape[0] + 20 * i + j, avgx + j * x, avgy + j * y, avgz + j * z)
                # new_point.append([avgx+j*x, avgy+j*y, avgz+j*z])
            #i += 50


        Polydata.SetPoints(test)

        delaunayFilter = vtk.vtkDelaunay2D()
        delaunayFilter.SetInputData(Polydata)
        delaunayFilter.SetTolerance(0.001)
        #delaunayFilter.SetAlpha(0.2)
        delaunayFilter.Update()

        o = delaunayFilter.GetOutput()
        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.SetInputData(o)

        smooth_loop = vtk.vtkLoopSubdivisionFilter()
        smooth_loop.SetNumberOfSubdivisions(3)
        smooth_loop.SetInputConnection(cleanPolyData.GetOutputPort())
        smooth_loop.Update()

        '''
        o2 = smooth_loop.GetOutput()
        cleanPolyData2 = vtk.vtkCleanPolyData()
        cleanPolyData2.SetInputData(o2)

        smooth_loop2 = vtk.vtkLoopSubdivisionFilter()
        smooth_loop2.SetNumberOfSubdivisions(3)
        smooth_loop2.SetInputConnection(cleanPolyData2.GetOutputPort())
        smooth_loop2.Update()
        '''
        #result = smooth_loop2.GetOutput()
        result = smooth_loop.GetOutput()
        vertexes = [result.GetPoint(i) for i in range(result.GetNumberOfPoints())]
        pdata = result.GetPolys().GetData()
        values = [int(pdata.GetTuple1(i)) for i in range(pdata.GetNumberOfTuples())]
        triangles = []
        while values:
            n = values[0]  # number of points in the polygon
            triangles.append(values[1:n + 1])
            del values[0:n + 1]

        Point_cordinates2 = result.GetPoints().GetData()
        numpy_coordinates2 = numpy_support.vtk_to_numpy(Point_cordinates2)

        print("myarea ", self.calculateArea(triangles, numpy_coordinates2))


        self.createNewModelNode(smooth_loop.GetOutput(), "Delaunay2D")

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
