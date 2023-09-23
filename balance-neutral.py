# import required module
import os
import random

# assign directory
directory = './FERPlus/train/neutral'

# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        delete = random.randint(0, 1)
        if delete == 1:
            os.remove(f)
            print("remove " + f)

print("done")
