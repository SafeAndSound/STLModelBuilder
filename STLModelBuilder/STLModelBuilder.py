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

  def install_dependencies(): # Download packages
    os.system("pip install imageio")
    
  install_dependencies()

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

    inputModelFile = "C:\\Users\\cglab-210\\Desktop\\Hand\\input\\Hand.stl" # File location
    outputDir = "C:\\Users\\cglab-210\\Desktop\\Hand\\output" # Output tiff file in rasterization form
    outputVolumeLabelValue = 100
    outputVolumeSpacingMm = [0.5, 0.5, 0.5]
    outputVolumeMarginMm = [10.0, 10.0, 10.0]

    # Read model
    inputModel = slicer.util.loadModel(inputModelFile)

    # Determine output volume geometry and create a corresponding reference volume
    bounds = np.zeros(6)
    inputModel.GetBounds(bounds)
    imageData = vtk.vtkImageData()
    imageSize = [ int((bounds[axis*2+1]-bounds[axis*2]+outputVolumeMarginMm[axis]*2.0)/outputVolumeSpacingMm[axis]) for axis in range(3) ]
    imageOrigin = [ bounds[axis*2]-outputVolumeMarginMm[axis] for axis in range(3) ]
    imageData.SetDimensions(imageSize)
    imageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    imageData.GetPointData().GetScalars().Fill(0)
    referenceVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    referenceVolumeNode.SetOrigin(imageOrigin)
    referenceVolumeNode.SetSpacing(outputVolumeSpacingMm)
    referenceVolumeNode.SetAndObserveImageData(imageData)
    referenceVolumeNode.CreateDefaultDisplayNodes()

    # Convert model to labelmap
    seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    seg.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolumeNode)
    slicer.modules.segmentations.logic().ImportModelToSegmentationNode(inputModel, seg)
    seg.CreateBinaryLabelmapRepresentation()
    outputLabelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(seg, outputLabelmapVolumeNode, referenceVolumeNode)
    outputLabelmapVolumeArray = (slicer.util.arrayFromVolume(outputLabelmapVolumeNode) * outputVolumeLabelValue).astype('int8')

    # Write labelmap volume to series of TIFF files
    import imageio
    for i in range(len(outputLabelmapVolumeArray)):
        imageio.imwrite(f'{outputDir}/image_{i:03}.tiff', outputLabelmapVolumeArray[i])
        
    logging.info('Processing completed')

    return True


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
