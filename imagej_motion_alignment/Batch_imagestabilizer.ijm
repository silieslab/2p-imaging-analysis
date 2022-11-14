setBatchMode(true);

function evaluate(subdir) {	
	numberofslices = flist2.length;
	fname = dir+subdir;
        fname = replace(fname,"/","\\");
	cmdstring = "open=["+fname+"] number=numberofslices starting=1 increment=1 scale=100 file=[] or=[] sort";
	run("Image Sequence...", cmdstring);

	name=replace(subdir,"/","");
	selectWindow(name);
	run("Duplicate...", "title=temp duplicate range=1-slicenumber");
	selectWindow("temp");
	run("Gaussian Blur...", "sigma=2 stack");
	run("Image Stabilizer", "transformation=Translation maximum_pyramid_levels=1 template_update_coefficient=0.90 maximum_iterations=200 error_tolerance=0.0000001 log_transformation_coefficients");
	close();

	selectWindow(name);
	run("Image Stabilizer Log Applier", " ");

	savePath = fname;
	print("Saving here: "+ savePath);
	saveAs("TIFF... ", savePath+replace(subdir,"/","")+"_Ch2_reg.tif");
	close();
	selectWindow("Log");
	run("Close");
	selectWindow("temp.log");
	run("Close");
	
	
}
dir = getDirectory("Select base folder");
flist = getFileList(dir);
for (i=0; i<flist.length; i++) {
	if (endsWith(flist[i], "/"))
		flist2 = getFileList(dir+flist[i]);
		print("Number of files in folder: "+ flist2.length);
		print("Processing: " + flist[i]);
		evaluate(flist[i]);
}

selectWindow("Log"); 
run("Close");
