import xml.etree.ElementTree as ET
import os
import scipy.misc
from shapely import geometry
import numpy as np

phase = 'val'

r = []
subdirs = [x[0] for x in os.walk("objects/"+phase)]
for subdir in subdirs:
    files = next(os.walk(subdir))[2]
    if (len(files) > 0):
        for file in files:
            r.append(subdir + "/" + file)
r.sort()
number = 0            
for path in r:
    with open(path) as f:
        root = ET.fromstringlist(["<root>", f.read(), "</root>"])
        for child in root:
            image_path = "images/" + path[8:-4] + ".jpg"
            image = scipy.misc.imread(image_path)
            if child.tag == 'objects':
                points = []                
                for obj in child:
                    if obj.tag == 'class':
                        obj_class = int(obj.text)
                    elif obj.tag == 'polygon':
                        for point in obj:
                            points.append([int(point[0].text), int(point[1].text)])
                    elif obj.tag == 'bndbox':
                        xmin = int(obj[0].text)
                        xmax = int(obj[1].text) + 1
                        ymin = int(obj[2].text)
                        ymax = int(obj[3].text) + 1
                poly = geometry.Polygon(points)
                # sub_image = np.zeros((128, 128, 3))
                # for x in range(xmin,xmax):
                #     for y in range(ymin,ymax):
                #         if poly.contains(geometry.Point(x,y)):
                #             sub_image[y,x,:] = image[y,x,:]
                if xmax-xmin >= 32 and ymax-ymin >= 32:
#                    sub_image = image[ymin:ymax, xmin:xmax, :]
#                    width = max(xmax-xmin, 0)
#                    height = max(ymax - ymin, 0)
#                    sub_image = scipy.misc.imresize(sub_image, (height, width, 3))
#                    save_path = "augment/"+phase+"/"+str(number)+".jpg"
#                    scipy.misc.imsave("images/"+save_path, sub_image)
                    save_path = path[8:-4] + ".jpg"
                    print(save_path + " " + str(obj_class))
                number += 1
