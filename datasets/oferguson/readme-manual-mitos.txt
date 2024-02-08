Folder: Manual_Mitos
Created by Owen Ferguson 2024.02.06

This folder contains images of mitochondria in human dopaminergic neurons. The goal with these data is to use mitochondrial morphology as an indicator of Parkinson's Disease.

The subfolders names are setup in this format:
	Lxx_CellLine_DIVyy

	Where xx refers to the differentiation (batch) number
	CellLine identifies the mutations in the folder
	yy The age of the cells in the folder

	Each of the above subfolders contains further subfolders:
		INPUT
			Tiff images
			Ch0 = TH (Dopaminergic neuron stain) CONF {488}
			Ch1 = Tom20 (Mitochondria) CONF {635}
			Ch2 = Tom20 STED {635}

		MASKS
			Tiff images; 1 channel
			Binary (0, 255) masks made from manual annotations of the input 				images. These masks were made while blinded to the condition


The INPUT and MASKS folders will contain further subfolders that split the images according to condition or treatment:

	AIW: Healthy Control
	A53T: PD line

	2KO: Healthy Control
	3x: PD line

	**These lines are paired. Do not cross the pairs i.e. do not use 2KO as a control 	for A53T; don't use AIW as a control for 3x.

	I'm sorry, there's one more layer of subfolders:
		
		Rotenone: Immediate PD-associated stressor that affects mitochondria
		PFF: Less intense stressor that takes longer to affect mitos
		Untreated: Baseline conditions with no added treatments/stressors


**Where should I start!?**
I would compare the 2KO and 3x lines when untreated vs when stressed with rotenone first. The rotenone should be very obviously fragmented. 

I expect a huge difference between untreated and rotenone, but within each group, I don't expect the 2KO and 3x sub-groups to be that different. 
	


