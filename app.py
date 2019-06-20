from builtins import print
from EnhancedFacenet import EnhancedFacenet
from FaceDetector import FaceDetector
from augmenter import NightVisionEffect,HistogramEqualizationEffect,HsvHistogramEqualizationEffect
import cv2
import os
import numpy as np
from random import randint
from annoy import AnnoyIndex
from PIL import Image, ImageEnhance
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-per", "--percentage", type=int, default=15,
	help="percentage")
ap.add_argument("-src", "--source", required=True,
	help="source")
ap.add_argument("-dest", "--dest", required=True,
	help="destination")
args= vars(ap.parse_args())



def vect(fd,ef,path):
    img = cv2.imread(path)

    rects = fd.detect(img)

    max_rect = None
    max_area = 0

    for rect in rects:
        if rect.width() * rect.height() > max_area:
            max_area = rect.width() * rect.height()
            max_rect = rect

    max_rect

    face, vector = ef.alignAndEncode(img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), max_rect)
    return vector

def aug(src,dest,p_nve,p_rot,p_brig):
    val=randint(8,16)
    nve = NightVisionEffect(gamma=val/10)
    hhee = HsvHistogramEqualizationEffect()
    hee = HistogramEqualizationEffect()

    src = src.replace("\n", "")
    img = cv2.imread(src,cv2.IMREAD_COLOR)


    if p_nve==1:
        img = nve.run(img)
        cv2.imwrite(dest, img)

    if p_rot==1:
        img = img[:, ::-1]
        cv2.imwrite(dest, img)

    if p_brig==1:
        im = Image.open(src)
        enhancer = ImageEnhance.Brightness(im)
        rand=randint(7,14)
        rand=rand/10
        enhanced_im = enhancer.enhance(rand)
        enhanced_im.save(dest)

persons= dict()
tempDict= dict()

path = args["dest"]+'/'
path1 = args["source"]+'/'

vectorArr=[]
nameArr=[]
tempArr = []


# Augmentation code
######################

file = os.listdir(args["source"])
try:
    for f in file:
        src = path1 + f
        dest = args["dest"] + '/' + f
        dest = dest.replace("\n", "")
        a = randint(0, 1)
        b = randint(0, 1)
        c = randint(0, 1)
        if a == 1:
            b = 0
            c = 0
        if b == 1:
            c = 0
        aug(src, dest, a, b, c)
except  Exception as e:
    print(e)
#####################

# exporting embeddings code

fd = FaceDetector()
ef = EnhancedFacenet()


images = os.listdir(args["dest"])
images.sort()


for per in images:

    if per == '':
        break
    else:
        pr = per.split('_')
        str1 = ""
        i = 0
        while (i < len(pr)):
            if i != len(pr) - 1:
                str1 = str1 + pr[i] + " "

            i = i + 1
        str1 = str1.strip()
        str1 = str1.replace(' ', '_')
        if str1 not in persons:
            persons[str1] = [per]

        else:
            persons[str1].append(per)


for i in persons:
    tempArr.append(i)

    size=int((len(persons)*args["percentage"]/100)+3)

for k in range(0,size):
    num = randint(0,len(tempArr)-1)
    if tempArr[num] not in tempDict:
        tempDict[tempArr[num]]=persons[tempArr[num]]

for per in tempDict:
    try:
        for item in tempDict[per]:
            vector = vect(fd, ef, path + item)
            nameArr.append(path + item)
            vectorArr.append(vector)
    except Exception as e:
        print(per)
        print(e)


np.save('vectors', vectorArr)
np.save('persons',nameArr)



###########################################
##             ANNOY SIDE                ##
###########################################

file = np.load('vectors.npy')




f = 512
t = AnnoyIndex(f,metric="euclidean")  # Length of item vector that will be indexed
for i in range(len(file)):
    v=file[i]
    t.add_item(i, v)


t.build(10) # 10 trees
t.save('test.ann')


u = AnnoyIndex(f,metric="euclidean")
u.load('test.ann') # super fast, will just mmap the file


av= np.load('vectors.npy')

pr= np.load('persons.npy')


arr = u.get_nns_by_vector(av[0],3,include_distances=True)
for i in arr[0]:
    print(pr[i])
