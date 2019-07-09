import os
pt = '/Users/neelrawat/Documents/SUTD/Term8/AI/Week 7/PetImages/'

imgnames = []
label = []
for root, directories, filename in os.walk(os.path.join(pt, 'Dog')):
    print(root, directories, filename)
    for filenames in filename:
        v = os.path.join(root, filenames)
        if os.path.isfile(v):
            imgnames.append(os.path.join('Dog', filenames))
            label.append(0)
#print(os.listdir(os.path.join(pt, 'Dog')))
print(imgnames)
print(label)
