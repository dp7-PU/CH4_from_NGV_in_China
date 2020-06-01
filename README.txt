The repository consists following source data and codes for the manuscript "Methane Emissions from Natural Gas Vehicles in China".

1. Calculate_ER.py
2. 0610_10Hz_with_ratios_1s_final.csv
3. 0611_10Hz_with_ratios_1s_final.csv
4. 0612_10Hz_with_ratios_1s_final.csv
5. Bus_0610.mat
6. Bus_0611.mat
7. Bus_0612.mat
8. Figure_source_data.xlsx
9. Gaussian_puff.py
10. README.txt

All the files should be placed together.

"Calculate_ER.py" is the source code for calculation of enhancement ratios using.
The code was made to run with Python 2.7. Three packages are needed:

1. Numpy
2. Scipy
3. Pandas

These packages are included in Anaconda Python distribution which can be downloaded
freely from (https://www.anaconda.com/). The typical install time is 30 min.

The code has been tested on Windows 10, with an Anaconda2-2019.10-Windows-x86_64 
distribution and on MacOS High Sierra with an Anaconda2-2019.10-MacOSX-x86-64 
distribution.

The code loads file 2-4, which contain raw CH4, CO2, NH3 observations, and
pre-calculated enhancement ratios and determination coefficients (R^2), and file 5-
7, which contain start times and durations of NGV encounters during our field 
campaign. There are three major sections in the code. The first one calculates
enhancement ratios, and the typical run time is one day. The second and the third
sections can run without executing the first section. The second section identifies
plumes related to NGVs. The typical runtime for section 2 is 30 - 60 s. 
The third section calculates mean ERs and their uncertainty. The typical runtime 
is <10 s.  
 
The first section saves 10 Hz ER to csv files. The second and third sections 
outputs the sample-size weigthed mean ER and its uncertainty.

File 8 contains all the source data for the figures in the manuscript and the SI.

File 9 contains the code for the Random Walk Gaussian Puff Model.     
