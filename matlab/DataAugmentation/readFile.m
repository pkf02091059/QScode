%readFile('/media/F/train_data/clothes/train/')
function readFile(path)
warning off all;
classes = dir(path);
num = length(classes);
matlabpool 4;
for j = 3 : num
    disp(['Begin Class: ' num2str(j-2)]);
    tic
    class_name = classes(j).name;
    class_path = [path class_name '/'];
    if length(dir(class_path))<400
        DataAugmentation(class_path);
    end
    toc
end
matlabpool close;
end