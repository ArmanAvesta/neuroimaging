classdef PtDiffStats < handle
    %Takes in the diffusion measures of one tract in one patient
    %Properties:
    %   vec = 1*c vector of one diffusion measure of one tract in one
    %   petient; c is the number of points along the tract
    %   cts = CtsDiffStats object, containing stats of controls
    %   injPoints = 1*c vector of abnormal points along the tract compared to
    %   potholes and molehills of controls; 1 is above molehill, 0 is
    %   between pothole and molehill, -1 is below molehill
    %   injPoints2 = 1*c vector of abnormal points along the tract compared to RANGE of
    %   values in controls; 1 is above upper threshold, 0 is withing min
    %   and max of controls, -1 is below lower threshold
    %   injNum = number of injured points along the tract based on
    %   injPoints==-1
    %   injNum2 = number of injured points along the tract based on
    %   injPoints2==-1
    %   injPercent = percent of injury of the tract based on injNum/c
    %   InjPercent2 = percent of injury of the tract based on InjNum2/c
    
    
    properties
        vec
        cts
        injPoints
        injPoints2
        injNum
        injNum2
        injPercent
        injPercent2
    end
    
    methods
        function obj = PtDiffStats(ptVec, ctsObj)
            c1 = length(ptVec);
            c2= size(ctsObj.mat, 2);
            if c1 ~= c2
                error('Constructor:DimensionError', 'The number of points along the path does not match between the patient and the group of controls')
            end
            obj.vec = ptVec;
            obj.cts = ctsObj;
            obj.calcInjPoints
            obj.calcInjPoints2
            obj.calcInjNum
            obj.calcInjNum2
            obj.calcInjPercent
            obj.calcInjPercent2
        end
        
        function calcInjPoints(obj)
            obj.injPoints = zeros(1,length(obj.vec));
            obj.injPoints(obj.vec > obj.cts.mht) = 1;
            obj.injPoints(obj.vec < obj.cts.pht) = -1;
        end
        
        function calcInjPoints2(obj)
            obj.injPoints2 = zeros(1, length(obj.vec));
            obj.injPoints2(obj.vec > obj.cts.maximum) = 1;
            obj.injPoints2(obj.vec < obj.cts.minimum) = -1;
        end
        
        function calcInjNum(obj)
            obj.injNum = sum(obj.injPoints == -1);
        end
        
        function calcInjNum2(obj)
            obj.injNum2 = sum(obj.injPoints2 == -1);
        end
        
        function calcInjPercent(obj)
            obj.injPercent = mean(obj.injPoints == -1);
        end
        
        function calcInjPercent2(obj)
            obj.injPercent2 = mean(obj.injPoints2 == -1);
        end
        
        function plotStats(obj)
            close all; lineStyle = 'k'; lineWidth = 2;
            
            subplot(2,2,1); hold on;
            r = size(obj.cts.mat, 1);
            for i = 1:r
                plot(obj.cts.mat(i,:), 'b');
            end
            plot(obj.vec, 'r', 'LineWidth', lineWidth)
            xlabel('Points along tract')
            ylabel('Diff values of patient (red) and controls (blue)')
            title('Raw values of patients and controls')
            
            subplot(2,2,2); hold on;
            plot(obj.cts.uot, lineStyle, 'LineWidth', lineWidth);
            plot(obj.cts.lot, lineStyle, 'LineWidth', lineWidth);
            plot(obj.vec, 'r', 'LineWidth', lineWidth);
            xlabel('Points along tract')
            ylabel('Patient vals and outlier thresholds based on controls')
            title('Quality check: checking outliers based on control data')
            
            subplot(2,2,3); hold on;
            plot(obj.cts.maximum, lineStyle, 'LineWidth', lineWidth);
            plot(obj.cts.minimum, lineStyle, 'LineWidth', lineWidth);
            plot(obj.vec, 'r', 'LineWidth', lineWidth);
            xlabel('Points along tract')
            ylabel('Patient and Threshold Diff Values')
            title('Patient Diff Vals compared to range of control vals')
            
            subplot(2,2,4); hold on;
            plot(obj.cts.mht, lineStyle, 'LineWidth', lineWidth)
            plot(obj.cts.pht, lineStyle, 'LineWidth', lineWidth)
            plot(obj.vec, 'r', 'LineWidth', lineWidth)
            xlabel('Points along tract')
            ylabel('Patient and Threshold Diff Values')
            title('Patient Diff Vals compared to PH and MH of control vals')   
        end
        
        function saveInjPoints(obj, filename)
            fid = fopen(filename, 'w');
            fprintf(fid, '%1.0f\n', obj.injPoints);
            fclose(fid);
        end
            
    end
    
end

