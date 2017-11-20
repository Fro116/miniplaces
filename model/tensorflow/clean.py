import os
for k in range(10):
    name = "tmp" + str(k)
    os.system("tail " + name + " -n +2 | sed -r 's/\[|]//g' > filr"+str(k))
