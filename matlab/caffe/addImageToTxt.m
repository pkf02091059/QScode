%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Tutor: Hao ZHANG
%Date: 2015/2/2
%Function: write images'name and type into a txt, p is the path of images
%          and doc is the txt document. Attention: there are 2 level dir
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%example addImage('./train/','train.txt'); addImage('./val/','val.txt')
function addImage(p,doc)
ID = dir(p); % son directory
fid = fopen(doc,'w'); % open txt
%load info;
for i = 3 : length(ID)
    %className = num2str(i-2);
    className = ID(i).name;
    path = [p className '/']; 
    %label{i-2} = str2num(ID(i).name);
    images = dir([path '*.*g']);% images
    images1 = dir([path '*.*G']);
    for j = 1 : length(images)
        %name = ['/opt/caffe-master/data/MyData' p(2:end) ID(i).name '/' images(j).name];%for txt
        name = [className '/' images(j).name]; %for lmdb
        %name = [path images(j).name];
        class_name = num2str(i-3);
        fprintf(fid,'%s %s\n',name,class_name);
    end
    for j = 1 : length(images1)
        %name = ['/opt/caffe-master/data/MyData' p(2:end) ID(i).name '/' images(j).name];%for txt
        name = [className '/' images1(j).name]; %for lmdb
        %name = [path images(j).name];
        class_name = num2str(i-3);
        fprintf(fid,'%s %s\n',name,class_name);
    end
end
fclose(fid);
disp('finish!');
%save info.mat info label
