function [ result ] = sepVarsLast( varList, sepTxt )
%sepVarsLast takes in the list of variables and returns the list of
%variables ending in sepText
%   varList = cell array containing the names of variables (can be made
%   by who)
%   sepTxt = some characters that the variables of interest end in
%   result = cell array containing the variables of interest

varList2 = cell(length(varList));

for j = 1:length(varList2)
    varList2{j} = fliplr(varList{j});
end

result = varList(strncmp(varList2, sepTxt, length(sepTxt)));

end

