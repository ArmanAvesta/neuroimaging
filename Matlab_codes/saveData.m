function [] = saveData( filename, data )
%saveData takes in a vector and saves it in a text file
%   filename: name of the file (string with extension like .txt) to be
%   saved
%   data: vector name in MATLAB environment

fid = fopen(filename, 'w');
fprintf(fid, '%1.0f\n', data);
fclose(fid);


end