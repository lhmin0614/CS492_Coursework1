% Please run this script under the root folder

clearvars -except N;
close all;

% addpaths
addpath('./Q4, Q5 internal');
addpath('./Q4, Q5 external');
addpath('./Q4, Q5 external/libsvm-3.18/matlab');

% initialise external libraries
run('Q4, Q5 external/vlfeat-0.9.21/toolbox/vl_setup.m'); % vlfeat library
cd('Q4, Q5 external/libsvm-3.18/matlab'); % libsvm library
run('make');
cd('../../..');

% tested on Ubuntu 12.04, 64-bit, Intel® Core�?i7-3820 CPU @ 3.60GHz × 8 
