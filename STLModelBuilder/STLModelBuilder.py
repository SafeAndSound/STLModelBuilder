import os
import unittest
import vtk
import qt
import ctk
import slicer
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
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene( slicer.mrmlScene )
        self.inputSelector.setToolTip( "Pick the input to the algorithm." )
        parametersFormLayout.addRow("Input STL Model: ", self.inputSelector)

        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Start Processing")
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
        logic.run(self.inputSelector.currentNode())

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

    def run(self, inputSTL):
        """
        Run the actual algorithm
        """
        print("----Start Processing----")
        print("Start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")

        STL_points_array = slicer.util.arrayFromModelPoints(inputSTL)
        STL_poly_array = slicer.util.arrayFromModelPolyIds(inputSTL)

        point_count = len(STL_points_array)
        poly_count = len(STL_poly_array) // 4
        print("三角形數量:{0}".format(poly_count))

        np_poly_array = np.zeros((poly_count, 3))

        edge_hash = {}
        vertex_hash = {}
        poly_hash = [[] for i in range(poly_count)]

        #Build edge and vertex datastruct
        for poly_id in range(poly_count):
        	vertices = STL_poly_array[poly_id * 4 + 1 : poly_id * 4 + 4]
        	np_poly_array[poly_id] = np.array(vertices)

        	self.addEdge(poly_id, vertices[0], vertices[1], edge_hash)
        	self.addEdge(poly_id, vertices[1], vertices[2], edge_hash)
        	self.addEdge(poly_id, vertices[0], vertices[2], edge_hash)

        	self.addVertex(poly_id, vertices[0], vertex_hash)
        	self.addVertex(poly_id, vertices[1], vertex_hash)
        	self.addVertex(poly_id, vertices[2], vertex_hash)

        for edge in edge_hash.keys():
        	if len(edge_hash[edge]) > 2:
        		print("重邊, 頂點: {0}, {1}".format(edge[0], edge[1]))
        	elif len(edge_hash[edge]) < 2:
        		print("邊緣, 頂點: {0}, {1}".format(edge[0], edge[1]))
        	else:
        		self.addPoly(edge_hash[edge][0], edge_hash[edge][1], poly_hash)

        poly_group = [-1] * poly_count #-1 if not visit
        group_id = 0

        #Build graph with DFS
        for poly_id in range(poly_count):
        	if poly_group[poly_id] == -1:
        		self.buildGroup(poly_id, group_id, poly_hash, poly_group)
        		group_id += 1

        print(len(poly_group))

        slicer.util.arrayFromModelPointsModified(inputSTL)
        self.rebuildNormals(inputSTL)

        print("\n----Complete Processing----")
        print("Complete time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


        return True

    def addEdge(self, poly, vertex1, vertex2, edge_hash):
        if (vertex1 < vertex2):
            edge = (vertex1, vertex2)
        else:
            edge = (vertex2, vertex1)

        if edge not in edge_hash:
            edge_hash[edge] = []
        edge_hash[edge].append(poly)

    def addVertex(self, poly, vertex, vertex_hash):
        if vertex not in vertex_hash:
            vertex_hash[vertex] = []
        vertex_hash[vertex].append(poly)

    def addPoly(self, poly1, poly2, poly_hash):
        if poly1 not in poly_hash:
            poly_hash[poly1] = []
        if poly2 not in poly_hash:
            poly_hash[poly2] = []
        poly_hash[poly1].append(poly2)
        poly_hash[poly2].append(poly1)

    def buildGroup(self, poly, group, poly_connection, poly_group):
        if poly_group[poly] != -1:
            return

        poly_group[poly] = group
        for adj_poly in poly_connection[poly]:
            self.buildGroup(adj_poly, group, poly_connection, poly_group)

    def rebuildNormals(self, model):
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(model.GetPolyData())
        normals.ComputePointNormalsOn()
        normals.SplittingOn()
        normals.Update()

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
