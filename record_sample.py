import numpy as np
import os


# Record the coordinates of sample points at certain outer iteration
def record_sample_coord(samples, iteration, path):

    filename = os.path.join(path, "Coordinates of sample points at {}-th outer iteration".format(iteration))
    f = open(filename, "w+")
    for line in samples:
        for numerics in line:
            f.write("%f " % numerics)
        f.write("\n")
    f.close()
