
run("Close All");

HOME="/Volumes/home/home-local2/projects/SSL/results/detection-experiments/MAE_SMALL_STED";
TEMPLATES=newArray("template-factin-assemblies", "template-factin-spots", "template-factin-spine", "template-factin-axon-rings");
//TEMPLATES=newArray("template-factin-assemblies-jchabbert", "template-factin-spots-jchabbert", "template-factin-spine-jchabbert");

COLORMAPS=newArray("   Grays ", "cb orange ", "cb reddishpurple ", "cb skyblue ", "biop-Chartreuse ");
MINIMUMS=newArray(0., 0.65, 0.65, 0.6, 0.65);

FOLDER="Block";
//FOLDER="multistructure-factin/raw";

// Possible images
//"11_Block_DIV13_SMI31-STAR580_MAP2-STAR488_PhSTAR635_2.msr_STED640_Conf561_Conf488_merged.tif"
//"12_Block_10_SMI31-STAR580_MAP2-STAR488_PhSTAR635_5.msr_STED640_Conf561_Conf488_merged.tif"
//"12_Block_10_SMI31-STAR580_MAP2-STAR488_PhSTAR635_3.msr_STED640_Conf561_Conf488_merged.tif"
//"04_Block_SMI31-STAR580_MAP2-STAR488_PhSTAR635_11.msr_STED640_Conf561_Conf488_merged.tif"
//"04_Block_SMI31-STAR580_MAP2-STAR488_PhSTAR635_3.msr_STED640_Conf561_Conf488_merged.tif"
//"01_Block_SMI31-STAR580_MAP2-STAR488_PhSTAR635_4.msr_STED640_Conf561_Conf488_merged.tif"
//"01_Block_SMI31-STAR580_MAP2-STAR488_PhSTAR635_3.msr_STED640_Conf561_Conf488_merged.tif"
IMAGENAME="01_Block_SMI31-STAR580_MAP2-STAR488_PhSTAR635_3.msr_STED640_Conf561_Conf488_merged.tif";

// Possible images Julia's dataset
//"7-Test-MOVIOL_PH635-STED-NO-FRESH_n5_crop_0_0.tif"
//"7-Test-PBS-1-day_PH635-STED-8rep-n6_crop_0_0.tif"
//IMAGENAME="7-Test-MOVIOL_PH635-STED-NO-FRESH_n5_crop_0_0.tif";

for (i=0; i<TEMPLATES.length; i++){
	template=TEMPLATES[i];
	path=HOME+"/"+template+"/"+FOLDER+"/"+IMAGENAME;
	open(path);
	title=getTitle();
	
	run("Duplicate...", "title=" + template + " duplicate channels=2-2");
	
	if (i == TEMPLATES.length - 1){
		selectImage(title);
		run("Duplicate...", "title=image duplicate channels=1-1");
	}
	close(title);
}

combine_channels="c1=image ";
for (i=0; i<TEMPLATES.length; i++){
	template=TEMPLATES[i];
	index=i+2;
	combine_channels=combine_channels + "c" + index + "=" + template + " ";
}
combine_channels=combine_channels + "create"

run("Merge Channels...", combine_channels);

width=0;height=0;channels=0;slices=0;frames=0;
Stack.getDimensions(width, height, channels, slices, frames);
for (i=0; i<channels; i++) {
	Stack.setChannel(i+1);
	run(COLORMAPS[i]);
	
	resetMinAndMax();
	m=0;M=0;
	getMinAndMax(m, M);
	if (i==0){
		run("Enhance Contrast", "saturated=0.35");
//		setMinAndMax(0.0, M);
	} else {
		M=0.80;
		M=1.0;
		setMinAndMax(MINIMUMS[i], M);
	}
}