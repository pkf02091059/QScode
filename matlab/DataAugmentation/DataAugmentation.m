%DataAugmentation('./image/')
function DataAugmentation(path)
warning off all;
images = dir([path '*.jpg']);
num = length(images);
%matlabpool('open','local',4);
parfor i = 1 : num
    try
    image_name = images(i).name;
    image_path = [path image_name];
    im = imread(image_path);
    catch err
        disp([image_path ' has problem']);
        continue;
    end
    %pad
    try 
        size1 = randi([20 100])
        size2 = randi([20 100])
        newIm = padarray(im,[size1 size2])
        imwrite(newIm,[path 'pad_' num2str(1) '_' num2str(i) '.jpg']);
        newIm = padarray(im,[size2 size1])
        imwrite(newIm,[path 'pad_' num2str(2) '_' num2str(i) '.jpg']);
    end
    %noise
    try
    H = fspecial('gaussian',[3 3],rand()*0.5);
    newIm = imnoise(newIm,'gaussian',rand()*0.2);
    imwrite(newIm,[path 'noise_1_' num2str(i) '.jpg']);   
    end
    %rotate
    try
    angle = randi([5 25]);
    newIm = imrotate(im,angle,'crop');
    imwrite(newIm,[path 'rotate_' num2str(5) '_' num2str(i) '.jpg']);
    
    newIm = flip(newIm,2);
    imwrite(newIm,[path 'flip_1_' num2str(i) '.jpg']);
    
    newIm = imrotate(im,-angle,'crop');
    imwrite(newIm,[path 'rotate_' num2str(-5) '_' num2str(i) '.jpg']);
    
    newIm = imrotate(im,90,'crop');
    imwrite(newIm,[path 'rotate_' num2str(90) '_' num2str(i) '.jpg']);
    
    newIm = imrotate(im,180,'crop');
    imwrite(newIm,[path 'rotate_' num2str(180) '_' num2str(i) '.jpg']);
    
    end
    %pyramid sampling
    %newIm = impyramid(im,'reduce');
    %imwrite(newIm,[path 'pyramid_reduce_' num2str(i) '.jpg']);
    %newIm = impyramid(im,'expand');
    %imwrite(newIm,[path 'pyramid_expand_' num2str(i) '.jpg']);
   % try
    %translate
    %imwrite(newIm,[path 'translate_1_' num2str(i) '.jpg']);
%    newIm = imtranslate(im,[randint(1,1,[fix(size(im,2)*0.1) fix(size(im,1)*0.1)]) randint(1,1,[fix(size(im,2)*0.1) fix(size(im,1)*0.1)]) 0]);
%    imwrite(newIm,[path 'translate_2_' num2str(i) '.jpg']);
%    newIm = imtranslate(im,[-randint(1,1,[fix(size(im,2)*0.15) fix(size(im,1)*0.15)]) -randint(1,1,[fix(size(im,2)*0.15) fix(size(im,1)*0.15)]) 0]);
%    imwrite(newIm,[path 'translate_3_' num2str(i) '.jpg']);
%    newIm = imtranslate(im,[-randi([fix(size(im,1)*0.1) fix(size(im,1)*0.2)]) -randi([fix(size(im,2)*0.1) fix(size(im,2)*0.2)]) 0]);
%    imwrite(newIm,[path 'translate_4_' num2str(i) '.jpg']);
%    newIm = imtranslate(im,[0 randi([fix(size(im,2)*0.05) fix(size(im,2)*0.2)]) 0]);
%    imwrite(newIm,[path 'translate_5_' num2str(i) '.jpg']);
%    newIm = imtranslate(im,[randi([fix(size(im,1)*0.1) fix(size(im,1)*0.2)]) 0 0]);
%    imwrite(newIm,[path 'translate_6_' num2str(i) '.jpg']);
%end
    
    %warp
    try
      step = randi([10 20])/100;  
      angle = randi([10 30])/100;
      
    tform = affine2d([1 0 0; angle 1 0; 0 0 1]);
    newIm = imwarp(im,tform);
    %imwrite(newIm,[path 'warp_1_' num2str(i) '.jpg']);
    
    newIm = imcrop(newIm,[randi([1 fix(size(newIm,2)*step)]) randi([1 fix(size(newIm,1)*step)]) fix(size(newIm,2)*(1-step)) fix(size(newIm,1)*(1-step))]);
    imwrite(newIm,[path 'crop_1_' num2str(i) '.jpg']);
    
    newIm = flip(newIm,2);
    imwrite(newIm,[path 'flip_2_' num2str(i) '.jpg']);
    
    tform = affine2d([1 0 0; -angle 1 0; 0 0 1]);
    newIm = imwarp(im,tform);
    %imwrite(newIm,[path 'warp_2_' num2str(i) '.jpg']);
    
    
    newIm = imcrop(newIm,[randi([1 fix(size(newIm,2)*step)]) randi([1 fix(size(newIm,1)*step)]) fix(size(newIm,2)*(1-step)) fix(size(newIm,1)*(1-step))]);
    imwrite(newIm,[path 'crop_2_' num2str(i) '.jpg']);
    %lightnesstform = affine2d([1 0 0; .15 1 0; 0 0 1]);
    end
    try
    angle = randi([10 30])/100.0;
    tform = affine2d([1 angle 0; 0 1 0; 0 0 1]);
    newIm = imwarp(im,tform);
    imwrite(newIm,[path 'warp_3_' num2str(i) '.jpg']);
    
    scale = randi([80 100])/100;
    newIm = imcrop(newIm,[randi([1 fix(size(newIm,2)*step)]) randi([1 fix(size(newIm,1)*step)]) fix(size(newIm,2)*scale*(1-step)) fix(size(newIm,1)*scale*(1-step))]);
    imwrite(newIm,[path 'crop_3_' num2str(i) '.jpg']);
    
    tform = affine2d([1 -angle 0; 0 1 0; 0 0 1]);
    newIm = imwarp(im,tform);
    imwrite(newIm,[path 'warp_4_' num2str(i) '.jpg']);
    newIm = imcrop(newIm,[randi([1 fix(size(newIm,2)*step)]) randi([1 fix(size(newIm,1)*step)]) fix(size(newIm,2)*scale*(1-step)) fix(size(newIm,1)*scale*(1-step))]);
    imwrite(newIm,[path 'crop_4_' num2str(i) '.jpg']);
    newIm = flip(newIm,2);
    imwrite(newIm,[path 'flip_3_' num2str(i) '.jpg']);
    end
    try
        ran1 = randi([10 20])/100.0;
        ran2 = randi([10 20])/100.0;
        ran3 = randi([60 80])/100.0;
        ran4 = randi([60 80])/100.0;
        newIm = imadjust(im,[ran1 ran2 0; ran3 ran4 1],[]);
        imwrite(newIm,[path 'lightness_1_' num2str(i) '.jpg']);
        newIm = imadjust(im,[ran3/4.2 ran4/4.2 0; ran1*5 ran2*4 1],[]);
        imwrite(newIm,[path 'lightness_2_' num2str(i) '.jpg']);
    %blur
        Blur = fspecial('gaussian',6,8);
        newIm = imfilter(im,Blur,'symmetric','conv');
        imwrite(newIm,[path 'blur_' num2str(i) '.jpg']);
    end
    try
        newIm = imcolor(im,1);
        imwrite(newIm,[path 'color_1_' num2str(i) '.jpg']);
        newIm = imcolor(im,2);
        imwrite(newIm,[path 'color_2_' num2str(i) '.jpg']);
        newIm = imcolor(im,3);
        imwrite(newIm,[path 'color_3_' num2str(i) '.jpg']);
        newIm = imcolor(im);
        imwrite(newIm,[path 'color_4_' num2str(i) '.jpg']);
        newIm = flip(newIm,2);
        imwrite(newIm,[path 'flip_4_' num2str(i) '.jpg']);
    end
    
    newIm = flip(im,2);
    imwrite(newIm,[path 'flip_5_' num2str(i) '.jpg']);
    
%    scale = randi([130 160])/100;
%    step = randi([5 15])/100;
%    newIm = imresize(im,[size(im,1) size(im,2)*scale]);
%    newIm = imcrop(newIm,[randi([1 fix(size(newIm,2)*step)]) randi([1 fix(size(newIm,1)*step)]) fix(size(newIm,2)*scale/1.5*(1-step)) fix(size(newIm,1)*scale/1.5*(1-step))]);
%    imwrite(newIm,[path 'crop_resize_' num2str(i) '.jpg']);
    %catch err
    %    disp(['Error :' image_name]);
    %end
    %imshow(newIm);
end
disp('Finish one directory!');
%matlabpool close;
end
