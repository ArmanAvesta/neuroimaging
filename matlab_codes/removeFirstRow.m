clear
load data
varList = who; varList(65:end) = [];

index = '(:,1)';
delMark = '[]';

[n, ~] = size(varList);

for i = 1:n
    var = varList{i};
    eval(sprintf('%s%s = %s;', var, index, delMark));
end

clear index delMark i n var varList