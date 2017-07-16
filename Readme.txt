1) Requirements :	
	Note: Code successfully compiled and ran on the version mentioned in the parenthesis for each library.

	1) Python (2.7)
	2) Pandas (0.19.2)
	3) NumPy (1.11.3)
	4) SciPy (0.19.0)
	5) Scikit-Learn (0.18.1)

2) Running :

	cd DIRECTORY
	python ocr.py

	Output:
		The program will generate three different text files as output

		1) learned_w_XXX.txt Learned State Parameters		
		2) learned_t_XXX.txt Learned Transition Parameters		
		3) log_XXX.txt Log of the run which contains train and test accuracies, time taken for training and testing etc.

		(XXX correspongs to current data set size.)

3) Different tunable Parameters:

	1) C - Regularization Constant
		Its default hardcoded to 1000, can be changed at line 165 in ocr.py
	2) maxiter - Maximum number of iteratios for optimization algorithm
		Its default hardcoded to 1000, can be changed at line 166 in ocr.py
	3) samples_count - Size of the dataset
		Its default hardcoded to 6600, can be changed at line 183 in ocr.py

4) Different files and folders in this directory:		
	
	1) ocr.py - Implementation of the model
	2) Letter.csv - Dataset
	3) letter.names.txt - Fields names of the dataset
	4) XXX-iterations-results - Directory
		- Contains output files of experiments with maxiter parameter = XXX
		- There are results for around 10 different experiments for each of maxiter XXX
		- Number in the filenames of parameters corresponds to dataset size
		- Output files includes learned state parameters, transition parameters and logs
	5) Report.pdf - Report
