classdef TractStats < handle
    %TractStat class contains the properties and methods to work with all
    %diffusion measures of one tract in a group of subjects
    %Properties:
    %   name = name of the tract
    %   FA = DiffStat object containing statistics of fractional anisotropy
    %   of one tract in a group of subjects
    %   MD = DiffStat object statistics of mean diffusivity
    %   RD = DiffStat ojbect statistics of radial diffusivity
    %   AD = DiffStat object statistics of axial diffusivity
    %Methods:
    %   Constructor: takes in r*c matrices of diffusion
    %   measures (FA, MD, RD, AD) and constructs the TractStats object;
    %   each row of matrices is a subject and each column is a point along
    %   the tract; if just one argument is passed to the constructor, it
    %   just constructs obj.FA and sets other object properties as NaN

    
    properties
        name
        FA
        MD
        RD
        AD
    end
    
    methods
        function obj = TractStats(tractName, matFAs, matMDs, matRDs, matADs)
            switch nargin
                case 5
                    obj.name = tractName;
                    obj.FA = DiffStats(matFAs);
                    obj.MD = DiffStats(matMDs);
                    obj.RD = DiffStats(matRDs);
                    obj.AD = DiffStats(matADs);
                case 2
                    obj.name = tractName;
                    obj.FA = DiffStats(matFAs);
                    obj.MD = NaN;
                    obj.RD = NaN;
                    obj.AD = NaN;
                otherwise
                    obj.name = NaN;
                    obj.FA = NaN;
                    obj.MD = NaN;
                    obj.RD = NaN;
                    obj.AD = NaN;
            end
        end
        
    end
    
end

