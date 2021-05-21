cmake_minimum_required(VERSION 3.5)

project(STLModelBuilder)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/STLModelBuilder")
set(EXTENSION_CATEGORY "NCTU")
set(EXTENSION_CONTRIBUTORS "NCTU Computer Graphics Laboratory")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/STLModelBuilder.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/STLModelBuilder/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(STLModelBuilder)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
