function [ sig ] = tTest( cts, pts )
%Takes in the pathway matrix of controls and patients and returns the p-value of
%the groups difference along the pathway (point-by-point)
%   cts: c*n controls matrix, each row beign a subject and each column being a
%   point along the pathway
%   pts: p*n patients matrix, each row beign a subject and each column being a
%   point along the pathway
%   sig: 1*n row vector containing the p-value of the point-by-point comparison
%   between controls and patients

[~, n] = size(cts);

sig = ones(1,n);

for j = 1:n
    c1 = cts(:,j);
    c2 = pts(:,j);
    
    c1(isnan(c1)) = [];
    c2(isnan(c2)) = [];
    
    m1 = mean(c1);
    m2 = mean(c2);
    v1 = var(c1);
    v2 = var(c2);
    n1 = length(c1);
    n2 = length(c2);
    
    t = (m1 - m2)/sqrt(v1/n1 + v2/n2);
    dof = (v1/n1 + v2/n2)^2 / (v1^2 / (n1^2*(n1-1)) + v2^2/(n2^2*(n2-1)));
    
    sig(j) = cdf('t', t, dof);
end

end

