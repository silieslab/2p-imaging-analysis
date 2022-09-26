The following folder structure is recommended to store your raw data and subsequent files created during the analysis.
Ennumeration represent folders, subfolders and files inside. 

1. Chosen name for the experiment (e.g "Tm9_recordings" or "Tm9_luminances")

  1.1. "rawData" (fixed folder name)
    1.1.1 "alignedData" (fixed folder name)
      1.1.1.1 Folder with date+userID+flyID (e.g. "2021_09_30_seb_fly1", from the microscope computer)
        1.1.1.1.1 TSeries (subfolder with recorded TIFFs)
          1.1.1.1.1.1 Motion aligned TIFF stack (generated after motion correction)
          1.1.1.1.1.2 Corresponding stim_output file to this TSeries (e.g." _stimulus_output_2022_09_30_16_53_46")
          
  1.2 "analyzed_data" (fixed folder name)
    1.2.1 Stimtype (e.g. "LocalCircle_5secON_5sec_OFF_120deg_10sec", folder created by the code)
      1.2.1.1 Defined genotype (e.g."ExpLine", folder created by the code)
        1.2.1.1.1 Pickle files (files generated and saved during the analysis)
          
  1.3 "stimulus_type" (fixed folder name)
    1.3.1 Files of the different stimulus input files used during the experiment (.txt files, e.g. "LocalCircle_5secON_5sec_OFF_120deg_10sec.txt")
    
2. "data_save_vars.txt" (fixed name, text file that lists the variables that want to save in a pickle file)
3. "roi_save_vars.txt" (fixed name, text file that lists the variables that want to save in a pickle file)
