Folder: Inserts_Images_and_Masks
*Last update: 2024.01.23 OF

This folder contains images of mitochondria taken with STED and confocal microscopy
The cells were plated on inserts for easier visualization

We are looking specifically at a Tom20 staining in human dopaminergic neurons (Day 60)


SubFolders:
**NOTE: Each subfolder should be viewed as a separate experiment

CROPS
	Same input and masks as described below, but each image was cropped into 4 smaller pieces
	The input images were only cropped, no other changes
	Cropped masks had any confocal objects at the borders removed
	Cropped STED masks were removed if the parent confocal object was removed

L25_AIW_A53T
	Lxx refers to the differentiation number
	L25 was started before L26
	Differentiations serve as biological replicates for the same experiments
	(i.e. it is valid to compare the 3x mutation between L26 and L32)	

	AIW/A53T refers to the cell condition
	AIW is a healthy control
	A53T is a Parkinson's Disease mutation

L26_2KO_3x_PFF
	2KO/3x refers to the cell condition
	2KO is a healthy control
	3x is a Parkinson's Disease mutation
	This mutation seems to have less of an impact on mitos than A53T

	PFF means that pre-formed alpha-synuclein fibrils were added to the cells
	This should induce a very obvious fragmentation at the mito level
	This batch of PFF didn't work though, so these data can be ignored

L18_2KO_3x_DIV88
	Same conditions as above, but the neurons are 88 days old

Remainder of folders:
The rest of the folders will look identical to the ones explained above, but will 
have different Lxx values. These further folders are biological replicates of 
previous experiments. For example, L32_2KO_3x is a biological replicate of L26_2KO_3x.


Sub-sub Folders:
INPUT
	The original images of Tom20
	(Rabbit anti-Tom20 attached to a STAR635P secondary)
	Ch0 = Confocal
	Ch1 = STED
	16-bit

MASKS
	The foreground segmentation of the input images
	Segmented using MitoSegNalysis/Segmenter_notebook.ipynb
	Wavelet segmentation with different parameters for Confocal/STED
	
	Objects smaller than mitochondria have already been removed

	Ch0 = Confocal Mask
	Ch1 = STED Mask
	8-bit 0/255 



GENERAL ADVICE / TL;DR

There is little biological value in comparing across lines (e.g. 2KO should always be compared to 3x). Comparing 2KO to AIW or A53T will likely not benefit us.

The PFF samples didn't really work and can be ignored.

2KO/3x replicates: L26, L31, L32
AIW/A53T replicates: L25, L30, L32

The difference in AIW vs A53T should be much easier to see than 2KO vs 3x. 


 
	
	