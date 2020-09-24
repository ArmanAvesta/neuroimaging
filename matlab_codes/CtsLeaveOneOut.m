classdef CtsLeaveOneOut < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        mat
        injPoints
        injNum
        injPercent        
    end
    
    methods
        function obj = CtsLeaveOneOut(ctsObj)
            obj.mat = ctsObj.mat;
            obj.calcInjPoints;
            obj.calcInjNum;
            obj.calcInjPercent;
        end
        
        function calcInjPoints(obj)
            [r, c] = size(obj.mat);
            obj.injPoints = zeros(r,c);
            for i = 1:r
                ctVec = obj.mat(i,:);
                ctsLeaveOneOutMat = obj.mat;
                ctsLeaveOneOutMat(i,:) = [];
                ctsObj = CtDiffStats(ctsLeaveOneOutMat);
                ctAsPt = PtDiffStats(ctVec, ctsObj);
                obj.injPoints(i,:) = ctAsPt.injPoints;
            end
        end
        
        function calcInjNum(obj)
            r = size(obj.mat, 1);
            obj.injNum = zeros(r,1);
            for i = 1:r
                ctVec = obj.mat(i,:);
                ctsLeaveOneOutMat = obj.mat;
                ctsLeaveOneOutMat(i,:) = [];
                ctsObj = CtDiffStats(ctsLeaveOneOutMat);
                ctAsPt = PtDiffStats(ctVec, ctsObj);
                obj.injNum(i) = ctAsPt.injNum;
            end
        end
        
        function calcInjPercent(obj)
          r = size(obj.mat, 1);
            obj.injNum = zeros(r,1);
            for i = 1:r
                ctVec = obj.mat(i,:);
                ctsLeaveOneOutMat = obj.mat;
                ctsLeaveOneOutMat(i,:) = [];
                ctsObj = CtDiffStats(ctsLeaveOneOutMat);
                ctAsPt = PtDiffStats(ctVec, ctsObj);
                obj.injPercent(i) = ctAsPt.injPercent;
            end
        end  
    end
    
end

