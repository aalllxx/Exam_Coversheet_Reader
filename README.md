# Exam_Coversheet_Reader
Takes scans of exam cover sheets and converts bubbles to number grades.

-----------
## Description
To save time when grading a large class, entering grades on paper and scanning them into a database can save time. This code does that using bubbles which can be filled in.

Images are precisely cropped using bracket markings in each corner of the document. Regions of interest (ROIs) are identified and a number is extracted from each. This is written to a file.

There are three types of ROIs supported currently - 3x3 grids, 10x2 grids, and 10x3 grids. The 3x3 are intended for a three digit unique student ID which students will fill out when taking the exam. The 10x2 is intended to score an individual problem with value between 0-99. 10x3 is for a final grade and supports values up to 999.

Creating a template for a region of interest in handled in the program. If it detects a file named ```ROIs.csv``` it will read the ROIs from that. Otherwise it will assist the user in creating one using a GUI.

Data are output to a file called ```grades.csv``` with row containing a unique student ID each column a value associate with that student.

## Syntax
- ```extractBubbles(pathToImages)```
- ```extractBubbles()```  will prompt the user to input a directory
- Images should in ```.png``` format.
- ```grades.csv``` and ```slices.cvs``` will be written to the directory listed
- Number of ROIs will be inferred from ```ROIs.csv``` or asked when generating creating the ROIs
