#!/usr/bin/env python

import os, math, sys, glob, csv
import fnmatch, cv2
import cv2.cv as cv
import numpy as np
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

__author__ = "Alexander Feldman"
__copyright__ = "Copyright 2017, Alexander Feldman"
__license__ = "GNU GPLv3"
__email__ = "felday@brandeis.edu"

def driver():
	#get directory of coversheets to scan
	dirpath, writefile = getDir()
	#load the template coversheet and set regions of interest
	setSlices(dirpath)
	#print number of files to the user
	numFiles = len(fnmatch.filter(os.listdir(dirpath), '*.png'))
	print('{} images found in {}'.format(numFiles,os.path.abspath(dirpath)))

	#begin the to write output
	with open(writefile, 'wb') as csvfile:
		gradeWriter = csv.writer(csvfile, dialect='excel')
		header = []
		for i in range(numROIs3x3):
			header.append('3x3_' + str(i))
		for i in range(numROIs10x2):
			header.append('10x2_' + str(i))
		for i in range(numROIs10x3):
			header.append('10x3_' + str(i))
		gradeWriter.writerow(header)

		#loop over each coversheet, extracting bubbles
		curFile = 0;
		for file in glob.glob(os.path.join(dirpath,'*.png')):
			#read the file
			[img, scaleFactor] = reader(file)
			#find the 'corners' of the sheet
			skewedImg = getCorners(img, scaleFactor)
			if skewedImg != []:
				#get all scores in sheet
				scores = getBubbles(skewedImg)
				scores.append(file)
			else:
				print("Couldn't crop image " + file)
				scores = [-1]
				continue
			#write scores to file
			gradeWriter.writerow(scores)
			#increment counters
			curFile += 1
			update_progress(curFile/float(numFiles))
	
	#terminate
	print('\rAll cover sheets marked and written to {0}'.format(os.path.abspath(writefile)))

def getDir():
	dirpath = None
	#if directory is entered in command line use it, otherwise prompt user
	if len(sys.argv) > 1:
		if os.path.isdir(sys.argv[1]):
			dirpath = sys.argv[1]
	while dirpath == None:
		tempDir = raw_input('Enter directory containing cover sheets: ')
		if os.path.isdir(tempDir):
			dirpath=tempDir
		else:
			print('Not a valid directory')

	#create a writefile. If it exists, create one with a new name
	if os.path.exists(os.path.join(dirpath, 'grades.csv')):
		i=1
		while os.path.exists('{}/grades{}.csv'.format(dirpath,i)):
			i+=1
		writefile = '{}/grades{}.csv'.format(dirpath,i)
	else:
		writefile = dirpath + '/grades.csv'
	return [dirpath, writefile]

def setSlices(dirpath):
	#make ranges for each ROI global to avoid passing them back and forth
	global ROIs3x3, ROIs10x2, ROIs10x3
	global numROIs3x3, numROIs10x2, numROIs10x3
	#if a template for coversheet exists use it, otherwise create one
	if os.path.exists(os.path.join(dirpath,'ROIs.csv')):
		with open(os.path.join(dirpath,'ROIs.csv'), 'rb') as csvfile:
			try:
				line_reader = csv.reader(csvfile, dialect='excel')
				[numROIs3x3, numROIs10x2, numROIs10x3] = map(int,csvfile.next().strip().split(',')[1:4])
				ROIs3x3 = []
				for i in range(numROIs3x3):
					ROIs3x3.append(map(float,csvfile.next().strip().split(',')[1:5]))
				ROIs10x2 = []
				for i in range(numROIs10x2):
					ROIs10x2.append(map(float,csvfile.next().strip().split(',')[1:5]))
				ROIs10x3 = []
				for i in range(numROIs10x3):
					ROIs10x3.append(map(float,csvfile.next().strip().split(',')[1:5]))
			except Exception as e:
				print ('Could not load ROIs.')
	else:
		promptForSlices()
		#write the newly generated template to disk
		with open(os.path.join(dirpath, 'ROIs.csv'), 'wb') as csvfile:
			slicewriter = csv.writer(csvfile, dialect='excel')
			slicewriter.writerow(['Num 3x3 10x2 10x3', numROIs3x3, numROIs10x2, numROIs10x3])
			for i in range(numROIs3x3):
				slicewriter.writerow(['3x3_'+str(i),ROIs3x3[i][0],ROIs3x3[i][1],ROIs3x3[i][2],ROIs3x3[i][3]])
			for i in range(numROIs10x2):
				slicewriter.writerow(['10x2_'+str(i),ROIs10x2[i][0],ROIs10x2[i][1],ROIs10x2[i][2],ROIs10x2[i][3]])
			for i in range(numROIs10x3):
				slicewriter.writerow(['10x3_'+str(i),ROIs10x3[i][0],ROIs10x3[i][1],ROIs10x3[i][2],ROIs10x3[i][3]])

def promptForSlices():
	#initialize ROIs
	global ROIs3x3, ROIs10x2, ROIs10x3
	global numROIs3x3, numROIs10x2, numROIs10x3
	numROIs3x3,numROIs10x2,numROIs10x3 = [None,None,None]
	ROIs3x3, ROIs10x2, ROIs10x3 = [],[],[]

	#prompt user for number of each type of ROI
	while numROIs3x3 == None:
		temp = raw_input('Enter number of 3x3 ROIs: ')
		if str.isdigit(temp):
			numROIs3x3 = int(temp)
		else:
			print('Not a valid number')
	while numROIs10x2 == None:
		temp = raw_input('Enter number of 10x2 ROIs: ')
		if str.isdigit(temp):
			numROIs10x2 = int(temp)
		else:
			print('Not a valid number')
	while numROIs10x3 == None:
		temp = raw_input('Enter number of 10x3 ROIs: ')
		if str.isdigit(temp):
			numROIs10x3 = int(temp)
		else:
			print('Not a valid number')

	#needed for gui
	def onselect(eclick, erelease):
		global l, t, r, b
		l1, t1, r1, b1 = map(int, [eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata])
		#allow selections that aren't top left to bottom right.
		l = min(l1,r1)
		r = max(l1,r1)
		t = min(t1,b1)
		b = max(t1,b1)

	#prompt for a template cover sheet for selecting regions of interest
	filename = None
	while filename == None:
		tempfile = raw_input('Path to template exam cover: ')
		if os.path.exists(tempfile):
			filename=tempfile
		else:
			print('Not a valid file')

	#instruct user in how to select	ROIs
	print('To select a region, click and drag a box around that region. The bounding box will be reduced to the tightest non-white pixels. Close the GUI to proceed.')

	#find corners of template
	fig = plt.figure()
	ax = fig.add_subplot(111)
	im = cv2.imread(filename,0)
	arr = np.asarray(im)
	plt_image=plt.imshow(arr)
	rs=widgets.RectangleSelector(
	    ax, onselect, drawtype='box',
	    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
	print ('Select corners')
	plt.show()

	#improve the user boxes for a tight bounding box
	internalTop, internalBot, internalLeft, internalRight = getTightBB(im[t:b,l:r])
	internalTop+=t
	internalBot=b-(b-(internalBot+t))
	internalLeft+=l
	internalRight=r-(r-(internalRight+l))
	internalHeight = internalBot-internalTop
	internalWidth = internalRight-internalLeft

	#crop the original image to just corners
	im = im[internalTop:internalBot, internalLeft:internalRight]
	arr = np.asarray(im)
	height, width = im.shape
	height = float(height)
	width = float(width)

	#get ROIs for 	
	for i in range(numROIs3x3):
		print ('Select 3x3 ROI ' + str(i+1))
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt_image=plt.imshow(arr)
		rs=widgets.RectangleSelector(
	    ax, onselect, drawtype='box',
	    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
		plt.show()
		boxTop, boxBot, boxLeft, boxRight = getTightBB(im[t:b,l:r])
		boxTop += t
		boxBot = b-(b-(boxBot+t))
		boxLeft += l
		boxRight = r-(r-(boxRight+l))
		ROIs3x3.append(list([boxTop/height, boxBot/height, boxLeft/width, boxRight/width]))
	for i in range(numROIs10x2):
		print ('Select 10x2 ROI ' + str(i+1))
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt_image=plt.imshow(arr)
		rs=widgets.RectangleSelector(
	    ax, onselect, drawtype='box',
	    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
		plt.show()
		boxTop, boxBot, boxLeft, boxRight = getTightBB(im[t:b,l:r])
		boxTop += t
		boxBot = b-(b-(boxBot+t))
		boxLeft += l
		boxRight = r-(r-(boxRight+l))
		ROIs10x2.append(list([boxTop/height, boxBot/height, boxLeft/width, boxRight/width]))
	for i in range(numROIs10x3):
		print ('Select 10x3 ROI ' + str(i+1))
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt_image=plt.imshow(arr)
		rs=widgets.RectangleSelector(
	    ax, onselect, drawtype='box',
	    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
		plt.show()
		boxTop, boxBot, boxLeft, boxRight = getTightBB(im[t:b,l:r])
		boxTop += t
		boxBot = b-(b-(boxBot+t))
		boxLeft += l
		boxRight = r-(r-(boxRight+l))
		ROIs10x3.append(list([boxTop/height, boxBot/height, boxLeft/width, boxRight/width]))

def getTightBB(img):
	height, width = img.shape
	# top 
	top = None
	for i in range(height-1):
		if top == None:
			for j in range(width):
				if img[i,j] < 200: #threshold
					top = i
		else:
			break
	# left
	left = None
	for j in range(width-1):
		if left == None:
			for i in range(height):
				if img[i,j] < 200: #threshold
					left = j
		else:
			break
	# bottom 
	bottom = None
	for i in range(height-1,1,-1):
		if bottom == None:
			for j in range(width):
				if img[i,j] < 200: #threshold
					bottom = i
		else:
			break
	# right
	right = None
	for j in range(width-1,1,-1):
		if right == None:
			for i in range(height):
				if img[i,j] < 200: #threshold
					right = j
		else:
			break

	# cv2.imshow('f',cv2.resize(img[top:bottom, left:right], (0,0), fx=0.25, fy=0.25))
	# cv2.waitKey(0)

	return [top,bottom,left,right]

def update_progress(progress):
    sys.stdout.write('\r[{0}] {1}%'.format('#'*(int(progress*10)), progress*100))
    sys.stdout.flush()

def reader(url):
	read = cv2.imread(url,0)
	height,width=read.shape
	scaleFactor = width/2500.0

	kernelSize = int(scaleFactor*11) 
	if kernelSize % 2==0:
		kernelSize+=1

	blur = cv2.GaussianBlur(read,(kernelSize,kernelSize),0) #kernel is 15
	img = cv2.threshold(blur,220,255,cv2.THRESH_BINARY_INV)[1]
	
	return [img, scaleFactor]

def getCorners(img, scaleFactor):
	height,width = img.shape
	outline = np.zeros([height,width])
	outline[0:height/6,0:width/4]=img[0:height/6,0:width/4]
	outline[0:height/6,3*(width/4):width-1]=img[0:height/6,3*(width/4):width-1]
	outline[5*(height/6):height-1,0:width/4]=img[5*(height/6):height-1,0:width/4]
	outline[5*(height/6):height-1,3*(width/4):width-1]=img[5*(height/6):height-1,3*(width/4):width-1]

	#find connected components in thresholded image
	labeled, numLabels = label(outline, structure = [[1,1,1],[1,1,1],[1,1,1]])
	image = np.zeros(img.shape)

	sizeThreshold = scaleFactor**2 * 2152 * .75 #refine this later
	for i in range(1,numLabels+1):
		TF = np.array(labeled == i)
		size = np.ndarray.sum(TF)

		if size > sizeThreshold: #2152 is corner size at SF=1
			image = image + (TF * img)

	[tl,tr,bl,br] = corner_looper(image)

	#test if four corners are accurately found
	negativeDiag = math.sqrt((br[1]-tl[1])**2 + (br[0]-tl[0])**2)
	positiveDiag = math.sqrt((bl[1]-tr[1])**2 + (tr[0]-bl[0])**2)

	#if the diagnals are not approximately equal, skip it.
	#future versions will handle this error
	if (positiveDiag > negativeDiag * 1.05):
		return []
	elif (negativeDiag > positiveDiag * 1.05):
		return []

	newDims = (tr[0]-tl[0],bl[1]-tl[1])
	points1 = np.float32([tl,tr,bl,br])
	points2 = np.float32([(0,0),(newDims[0]-1,0),(0,newDims[1]-1),(newDims[0]-1,newDims[1]-1)])
	M = cv2.getPerspectiveTransform(points1,points2)
	skewedImg = cv2.warpPerspective(img,M,newDims)
	skewedImg = cv2.resize(skewedImg, (2278, 3074))
	return skewedImg

def corner_looper(image):
	height, width = image.shape

	tl,tr,bl,br = [],[],[],[]

	#find corners in the skeleton image. Only search in corner of image
	for i in range(height/6):
		for j in range(width/4):
			if image[i,j]!=0:
				if tl==[] or sum(tl)>i+j:
					tl = [j,i] #top left
	for i in range(height/6):
		for j in range(width-1, 3*width/4, -1):
			if image[i,j]!=0:
				if tr==[] or ((tr[1]-tr[0])>(i-j)):
					tr = [j,i] #top right
	for i in range(height-1, 5*height/6, -1):
		for j in range(width/4):
			if image[i,j]!=0:
				if bl==[] or ((bl[0]-bl[1])>(j-i)):
					bl = [j,i] #bot left
	for i in range(height-1, 5*height/6, -1):
		for j in range(width-1, 3*width/4, -1):
			if image[i,j]!=0:
				if br==[] or sum(br)<(i+j):
					br = [j,i] #bottom right
	return [tl,tr,bl,br]

def getBubbles(skewedImg):
	global ROIs3x3, ROIs10x2, ROIs10x3
	global numROIs3x3, numROIs10x2, numROIs10x3
	height, width = skewedImg.shape
	output = []

	#loop through each ROI and get results
	for i in range(numROIs3x3):
		slice = getSlice(height, width, ROIs3x3[i])
		output.append(numberFrom3x3(skewedImg[slice[0]:slice[1],slice[2]:slice[3]]))
	for i in range(numROIs10x2):
		slice = getSlice(height, width, ROIs10x2[i])
		output.append(numberFrom10x2(skewedImg[slice[0]:slice[1],slice[2]:slice[3]]))
	for i in range(numROIs10x3):
		slice = getSlice(height, width, ROIs10x3[i])
		output.append(numberFrom10x3(skewedImg[slice[0]:slice[1],slice[2]:slice[3]]))
	return output

def getSlice(height, width, fraction):
	return ([int(math.floor(height*fraction[0])),int(math.ceil(height*fraction[1])),
		int(math.floor(width*fraction[2])),int(math.ceil(width*fraction[3]))])

def topBubbleVetical(img, numBubbles):
	# To remove number from empty bubbles, uncomment this section.
	# outline = np.zeros(img.shape, dtype=np.uint8)
	# labeled, numLabels = label(img, structure = [[1,1,1],[1,1,1],[1,1,1]])
	# for i in range(1,numLabels+1):
	# 	TF = np.array(labeled == i)
	# 	size = np.ndarray.sum(TF)
	# 	if size > 500: #optimize this number
	# 		outline = outline + (TF * img)
	# img = outline

	#get the average color of each bubble
	accum = []
	height, width = img.shape
	increment = int(height/numBubbles)
	for cir in range(numBubbles):
		accum.append(avgColor(img[cir*increment:(cir+1)*increment,:]))

	#select the top bubble in the column	
	result = [-1,-1]
	bubbleThreshold = .55 #optimize this number
	for i in range(numBubbles):
		if accum[i] > bubbleThreshold and accum[i] > result[1]:
			result = [i, accum[i]]
	return result

def numberFrom10x3(img):
	#separate the columns
	height, width = img.shape
	firstCol = img[:,0:int(width*.32)]
	secondCol = img[:,int(width*.33):int(width*.67)]
	thirdCol = img[:,int(width*.68):width-1]

	#get max value in each column
	maxFirst = topBubbleVetical(firstCol, 10)
	maxSecond = topBubbleVetical(secondCol, 10)
	maxThird = topBubbleVetical(thirdCol, 10)

	#return number from columns
	if maxFirst[0] != -1 and maxSecond[0] != -1:
		return 100*maxFirst[0]+10*maxSecond[0]+maxThird[0]
	else: #missing info
		return -1

def numberFrom10x2(img):
	#separate the columns
	height, width = img.shape
	firstCol = img[:,0:int(width*.47)]
	secondCol = img[:,int(width*.52):width]

	#get max value in each column
	maxFirst = topBubbleVetical(firstCol, 10)
	maxSecond = topBubbleVetical(secondCol, 10)

	#return number from columns.
	if maxFirst[0] != -1 and maxSecond[0] != -1:
		return 10*maxFirst[0]+maxSecond[0]
	else: #missing info
		return -1

def numberFrom3x3(img):
	#separate the columns	
	height, width = img.shape
	firstCol = img[:,0:int(width*.32)]
	secondCol = img[:,int(width*.33):int(width*.67)]
	thirdCol = img[:,int(width*.68):width-1]
	
	#get max value in each column
	maxFirst = topBubbleVetical(firstCol, 3)
	maxSecond = topBubbleVetical(secondCol, 3)
	maxThird = topBubbleVetical(thirdCol, 3)

	#return number from columns. 3x3 handles blank columns, other ROIs do not.
	if maxFirst[1] > maxSecond[1] and maxFirst[1] > maxThird[1]:
		return maxFirst[0]*3+1
	elif maxSecond[1] > maxFirst[1] and maxSecond[1] > maxThird[1]:
		return maxSecond[0]*3+2
	elif maxThird[1] > maxFirst[1] and maxThird[1] > maxSecond[1]:
		return maxThird[0]*3+3
	else: #all equal
		return -1;

def avgColor(img):
	#return the average pixel value in an image
	height, width = img.shape

	pixelCount = 0.0
	pixelSum = 0.0

	for i in range(height):
		for j in range(width):
			pixelCount +=1
			pixelSum += img[i,j]
	return (pixelSum/255)/pixelCount

driver()
