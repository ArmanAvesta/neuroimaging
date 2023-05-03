classdef PtsDiffStats < handle
    %Takes in the diffusion measures of one tract in all patients
    %Properties:
    %   mat = r1*c vector of one diffusion measure of one tract in all
    %   patients; r is the number of patients and c is the number of points
    %   along the tract
    %   cts = CtsDiffStats object, containing stats of controls
    %   
    
    properties
        mat
        cts
        injPoints
        injNum
        injPercent
    end
    
    methods
        function obj = PtsDiffStats(ptMat, ctsObj)
            c1 = size(ptMat,2);
            c2 = size(ctsObj.mat, 2);
            if c1 ~= c2
                error('Constructor:DimensionError', 'The number of points along the tract does not match between patietns and controls')
            end
            obj.mat = ptMat;
            obj.cts = ctsObj;
            obj.calcInjPoints;
            obj.calcInjNum;
            obj.calcInjPercent;
        end
        
        function calcInjPoints(obj)
            [r, c] = size(obj.mat);
            obj.injPoints = zeros(r,c);            
            for i = 1:r
                pt = PtDiffStats(obj.mat(i,:), obj.cts);
                obj.injPoints(i,:) = pt.injPoints;
            end
        end
        
        function calcInjNum(obj)
            r = size(obj.mat, 1);
            obj.injNum = zeros(r,1);
            for i = 1:r
                pt = PtDiffStats(obj.mat(i,:), obj.cts);
                obj.injNum(i) = pt.injNum;
            end
        end
        
        function calcInjPercent(obj)
            r = size(obj.mat, 1);
            obj.injPercent = zeros(r,1);
            for i = 1:r
                pt = PtDiffStats(obj.mat(i,:), obj.cts);
                obj.injPercent(i) = pt.injPercent;
            end
        end
        
        function plotStats(obj, nthCt)
            close all; lineStyle = 'k'; lineWidth = 2;
            
            subplot(2,2,1); hold on;            
            r = size(obj.mat, 1);
            for i = 1:r
                plot(obj.mat(i,:), 'r');
            end
            r = size(obj.cts.mat, 1);
            for i = 1:r
                plot(obj.cts.mat(i,:), 'b');
            end
            xlabel('Points along tract')
            ylabel('Diff Meas')
            title('Raw values of patients and controls')
            
            subplot(2,2,2); hold on;
            r = size(obj.mat, 1);
            for i = 1:r
                plot(obj.mat(i,:), 'r');
            end
            plot(obj.cts.uot, lineStyle, 'LineWidth', lineWidth);
            plot(obj.cts.lot, lineStyle, 'LineWidth', lineWidth);
            xlabel('Points along tract')
            ylabel('Diff Meas & Outlier Thresholds')
            title('Quality check: Outlier check')
            
            subplot(2,2,3); hold on;
            r = size(obj.mat, 1);
            for i = 1:r
                plot(obj.mat(i,:), 'r');
            end
            plot(obj.cts.maximum, lineStyle, 'LineWidth', lineWidth);
            plot(obj.cts.minimum, lineStyle, 'LineWidth', lineWidth);
            xlabel('Points along tract')
            ylabel('Diff Meas & Abnormal Thresholds')
            title('Patient Diff Vals compared to range of control vals')
            
            subplot(2,2,4); hold on;
            r = size(obj.mat, 1);
            for i = 1:r
                plot(obj.mat(i,:), 'r');
            end
            plot(obj.cts.mht, lineStyle, 'LineWidth', lineWidth)
            plot(obj.cts.pht, lineStyle, 'LineWidth', lineWidth)
            xlabel('Points along tract')
            ylabel('Diff Meas & Abnormal Thresholds')
            title('Patients Diff Vals compared to PH and MH of control vals')
            
            if nargin == 2
                for sbplt = 2:4
                    subplot(2,2,sbplt)
                    plot(obj.cts.mat(nthCt,:), 'b', 'LineWidth', lineWidth)
                end
            end
        end
        
        function plotPt(obj, nthPt)
            pt = PtDiffStats(obj.mat(nthPt,:), obj.cts);
            pt.plotStats;
        end
        
    end
    
end

