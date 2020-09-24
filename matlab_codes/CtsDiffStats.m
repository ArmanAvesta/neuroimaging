classdef CtsDiffStats < handle
    %DiffStat class contains the properties and methods to work with one
    %single diffusion measure (like FA) in one white matter tract in a
    %group of subjects
    %Properties:
    %   mat = r*c matrix of one diffusion measure of one tract in all subjects;
    %   rows being subjects and columns being points along the tract
    %   mn = 1*c vector of means of the diff meas in all sujbects for each point along the
    %   tract
    %   sd = 1*c vector of SD of the diff meas in all subjects for each
    %   point along the tract
    %   q1 = 1*c vector of lower quartiles of the diff meas along the tract
    %   q2 = 1*c vector of medians of the diff meas along the tract
    %   q3 = 1*c vector of upper quartiles of the diff meas along the tract
    %   lot = 1*c vector of lower outlier thresholds along the tract
    %   uot = 1*c vector of upper outlier thresholds along the tract
    %   pht = 1*c vector of pothole thresholds along the tract
    %   mht = 1*c vector of molehill thresholds along the tract
    %   maximum = 1*c vector of max values along the pathway
    %   minimum = 1*c vector of min values along the pathway
    %   K = coefficient of outlier calculation; default = 1.5
    %   Z = coefficient of pothole & molehill calculation; default = 1.96
    %Methods:
    %   Constructor: takes in r*c matrix of one diffusion measure,
    %   each row being a subject and each column being a point along the
    %   tract, and constructs the DiffStats object for one diffusion
    %   measure of one tract
    
    properties
        matOrig
        mat
        mn
        sd
        q1
        q2
        q3
        lot
        uot
        pht
        mht    
        maximum
        minimum
        K = 1.5;
        Z = 1.96;
    end
    
    methods
        function obj = CtsDiffStats(diffMeasMatrx)
            obj.matOrig = diffMeasMatrx;
            obj.calcQuartiles;            
            obj.calcOutlierThresholds;
            obj.removeOutliers;
            obj.calcMean;
            obj.calcStd;
            obj.calcNormalBandSD;
            obj.calcNormalBandRange;
        end        
        
        function calcQuartiles(obj)
            c = size(obj.matOrig, 2);
            obj.q1 = zeros(1,c);
            obj.q2 = zeros(1,c);
            obj.q3 = zeros(1,c);
            for j = 1:c
                col = obj.matOrig(:,j);
                col(isnan(col)) = [];
                qq2 = median(col);
                lowerhalf = col(col < qq2);
                upperhalf = col(col >= qq2);
                qq1 = median(lowerhalf);
                qq3 = median(upperhalf);
                
                obj.q1(j) = qq1;
                obj.q2(j) = qq2;
                obj.q3(j) = qq3;
            end
        end
        
        function calcOutlierThresholds(obj)
            obj.lot = obj.q1 - obj.K * (obj.q3 - obj.q1);
            obj.uot = obj.q3 + obj.K * (obj.q3 - obj.q1);
        end
        
        function removeOutliers(obj)
            r = size(obj.matOrig, 1);
            lowMat = ones(r,1) * obj.lot;
            upMat = ones(r,1) * obj.uot;
            obj.mat = obj.matOrig;
            obj.mat(obj.mat > upMat) = NaN;
            obj.mat(obj.mat < lowMat) = NaN;
        end
        
        function calcMean(obj)
            c = size(obj.mat, 2);
            obj.mn = zeros(1,c);
            for j = 1:c
                col = obj.mat(:,j);
                col(isnan(col)) = [];
                obj.mn(j) = mean(col);
            end
        end
        
        function calcStd(obj)
            c = size(obj.mat, 2);
            obj.sd = zeros(1,c);
            for j = 1:c
                col = obj.mat(:,j);
                col(isnan(col)) = [];
                obj.sd(j) = std(col);
            end
        end
        
        function calcNormalBandSD(obj)
            obj.pht = obj.mn - obj.Z * obj.sd;
            obj.mht = obj.mn + obj.Z * obj.sd;
        end
        
        function calcNormalBandRange(obj)
            c = size(obj.mat, 2);
            obj.maximum = zeros(1,c);
            for j = 1:c
                col = obj.mat(:,j);
                col(isnan(col)) = [];
                obj.maximum(j) = max(col);
                obj.minimum(j) = min(col);
            end
        end
        
        function plotStats(obj, nthCt)            
            close all;
            r = size(obj.mat, 1);
            
            subplot(2,2,1); hold on;
            for i = 1:r
                plot(obj.mat(i,:), 'g');
            end
            xlabel('Points along tract')
            ylabel('Diff value of subjects')
            title('Raw Diff Values')
            
            subplot(2,2,2); hold on;
            for i = 1:r
                plot(obj.mat(i,:), 'g');
            end
            plot(obj.uot, 'k--');
            plot(obj.lot, 'k--');
            xlabel('Points along tract')
            ylabel('Diff Outlier Thresholds')
            title('Quality check: Upper and Lower Outlier Thresholds')
            
            subplot(2,2,3); hold on;
            for i = 1:r
                plot(obj.mat(i,:), 'g')
            end
            plot(obj.maximum, 'b--')
            plot(obj.minimum, 'b--')
            xlabel('Points along tract')
            ylabel('Diff max and min values')
            title('Max and Min Diff Values')
            
            subplot(2,2,4); hold on;
            for i = 1:r
                plot(obj.mat(i,:), 'g')
            end
            plot(obj.mht, 'b--')
            plot(obj.pht, 'b--')
            xlabel('Points along tract')
            ylabel('Diff Pothole and Molehill Thresholds')
            title('Pothole and Molehill Threshold Values') 
            
            if nargin == 2
                for sbplt = 1:4
                    subplot(2,2,sbplt)
                    plot(obj.mat(nthCt,:), 'r')
                end
            end
            
        end
        
        function plotOrig(obj, nthCt)            
            close all;
            r = size(obj.matOrig, 1);
            
            subplot(2,2,1); hold on;
            for i = 1:r
                plot(obj.matOrig(i,:), 'g');
            end
            xlabel('Points along tract')
            ylabel('Diff value of subjects')
            title('Raw Diff Values')
            
            subplot(2,2,2); hold on;
            for i = 1:r
                plot(obj.matOrig(i,:), 'g');
            end
            plot(obj.uot, 'k--');
            plot(obj.lot, 'k--');
            xlabel('Points along tract')
            ylabel('Diff Outlier Thresholds')
            title('Quality check: Upper and Lower Outlier Thresholds')
            
            subplot(2,2,3); hold on;
            for i = 1:r
                plot(obj.matOrig(i,:), 'g')
            end
            plot(obj.maximum, 'b--')
            plot(obj.minimum, 'b--')
            xlabel('Points along tract')
            ylabel('Diff max and min values')
            title('Max and Min Diff Values')
            
            subplot(2,2,4); hold on;
            for i = 1:r
                plot(obj.matOrig(i,:), 'g')
            end
            plot(obj.mht, 'b--')
            plot(obj.pht, 'b--')
            xlabel('Points along tract')
            ylabel('Diff Pothole and Molehill Thresholds')
            title('Pothole and Molehill Threshold Values') 
            
            if nargin == 2
                for sbplt = 1:4
                    subplot(2,2,sbplt)
                    plot(obj.matOrig(nthCt,:), 'r')
                end
            end
            
        end        
                
    end
    
end

