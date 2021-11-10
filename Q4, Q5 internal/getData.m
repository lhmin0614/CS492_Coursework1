function [ data_train, data_query ] = getData( MODE , numBins)
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle
%   4. Caltech 101

showImg = 0; % Show training & testing images and their image feature vector (histogram representation)

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
    case 'Toy_Gaussian' % Gaussian distributed 2D points
        %rand('state', 0);
        %randn('state', 0);
        N= 150;
        D= 2;
        
        cov1 = randi(4);
        cov2 = randi(4);
        cov3 = randi(4);
        
        X1 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov1 0;0 cov1]);
        X2 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov2 0;0 cov2]);
        X3 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov3 0;0 cov3]);
        
        X= real([X1; X2; X3]);
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Spiral' % Spiral (from Karpathy's matlab toolbox)
        
        N= 50;
        t = linspace(0.5, 2*pi, N);
        x = t.*cos(t);
        y = t.*sin(t);
        
        t = linspace(0.5, 2*pi, N);
        x2 = t.*cos(t+2);
        y2 = t.*sin(t+2);
        
        t = linspace(0.5, 2*pi, N);
        x3 = t.*cos(t+4);
        y3 = t.*sin(t+4);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Circle' % Circle
        
        N= 50;
        t = linspace(0, 2*pi, N);
        r = 0.4
        x = r*cos(t);
        y = r*sin(t);
        
        r = 0.8
        t = linspace(0, 2*pi, N);
        x2 = r*cos(t);
        y2 = r*sin(t);
        
        r = 1.2;
        t = linspace(0, 2*pi, N);
        x3 = r*cos(t);
        y3 = r*sin(t);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Caltech' % Caltech dataset
        close all;
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        %disp(classList)
        classList(strncmp({classList.name}, '.DS', 1)) = [];
        %disp(classList)
        classList = {classList(1:end).name}; % 10 classes
        
        %disp(classList)
        
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            %disp(subFolderName)
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            
            imgIdx{c} = randperm(length(imgList));
                %disp(length(imgList))
                %disp(size(imgSel))
            imgIdx_tr = imgIdx{c}(1:imgSel(1));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            
            %desc_tr = 3차원 descriptor 행렬
            end
        end
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method'
        % extract 10e4 columns from cat(2, desc_tr{:})
        desc_sel = double(vl_colsubset(cat(2,desc_tr{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering
        %10e4 == 100000
        % K-means clustering
        %numBins = 128; % for instance
        %disp(desc_tr{3, 4})
        %disp(size(desc_tr{3, 4}))
        
        % write your own codes here
        % ...
        stream = RandStream('mlfg6331_64');  % Random number stream
        options = statset('UseParallel',1,'UseSubstreams',1, 'Streams',stream);
        tic; % Start stopwatch timer
        [idx,C,sumd,D] = kmeans(transpose(desc_sel),numBins,'Options',options,'MaxIter',10000,'Display','final','Replicates',10);
        %C = 256 * 128 행렬, (256개의 cluster 값)
        toc;
        disp(size(idx))
        %disp(D/100000)
        disp(size(C))
        
        disp('Encoding Images...')
        % Vector Quantisation
        
        % write your own codes here
        % ...
        mi = inf;

        for i = 1:10
            %disp(i)
            for j = 1:15     
                h = zeros(numBins, 1);
                n = size(desc_tr{i, j}, 2);
                descs = desc_tr{i, j};
                %disp(j)
                for k = 1:n
                    desc = descs(:,k);
                    desc = double(desc);
                    %disp(desc)
                    %disp(C(1,:))
                    for l = 1: numBins
                        d = norm(desc - transpose(C(l,:)));
                        if d < mi
                            mi = d;
                            mark = l;
                        end
                    end
                    mi = inf;
                    h(mark, 1) = h(mark, 1) + 1;
                end
                if i == 1 && j == 1
                    trhis = h;
                else
                    trhis = cat(2, trhis, h);
                end
            
            end
            
        end
        index = [ones(15, 1); ones(15, 1)*2; ones(15, 1)*3; ones(15, 1)*4; ones(15, 1)*5; ones(15, 1)*6; ones(15, 1)*7; ones(15, 1)*8; ones(15, 1)*9; ones(15, 1)*10];
        data_train = [trhis' index];
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
end

switch MODE
    case 'Caltech'
        if showImg
        figure('Units','normalized','Position',[.05 .1 .4 .9]);
        suptitle('Test image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
            
            end
        end
        %suptitle('Testing image samples');
%                 if showImg
%             figure('Units','normalized','Position',[.5 .1 .4 .9]);
%         suptitle('Testing image representations: 256-D histograms');
%         end

        % Quantisation
        
        % write your own codes here
        % ...
                mi = inf;

        for i = 1:10
            
            %disp(i)
            for j = 1:15  
                h = zeros(numBins, 1);
                n = size(desc_te{i, j}, 2);
                descs = desc_te{i, j};
                %disp(j)
                for k = 1:n
                    desc = descs(:,k);
                    desc = double(desc);
                    for l = 1: numBins
                        d = norm(desc - transpose(C(l,:)));
                        if d < mi
                            mi = d;
                            mark = l;
                        end
                    end
                    mi = inf;
                    h(mark, 1) = h(mark, 1) + 1;
                end
                if i == 1 && j == 1
                    tehis = h;
                else
                    tehis = cat(2, tehis, h);
                end
            end

            
        end
        data_query = [tehis' index];
        
    otherwise % Dense point for 2D toy data
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end
end

