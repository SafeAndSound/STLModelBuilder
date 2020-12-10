from PIL import Image
import numpy as np

class Obj:

    def loadObjFileWithPath(path):
        verticeId = 0
        current_deleted = 0

        with open(path) as file:
            points = []
            polys = []

            while 1:
                line = file.readline()
                if not line:
                    break

                strs = line.split(" ")

                if strs[0] == "v":
                    points.append((float(strs[1]), float(strs[2]), float(strs[3])))
                    verticeId += 1
                
                elif strs[0] == "vt":
                    break
                
                elif strs[0] == "f":
                    break

                elif strs[0] == "usemtl":
                    break
        
        self.pointData = np.array(points)
    
    def loadTextureFileWithPath(path):