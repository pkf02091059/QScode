function newIm = imcolor(im,index)
%im(:,:,index) = im(:,:,index)*randi([5 10])*0.1+randi([5 50]);
if nargin == 2
    im(:,:,index) = im(:,:,index)+randi([-30 30]);
else
    im(:,:,1) = im(:,:,1)+randi([-30 30]);
    im(:,:,2) = im(:,:,2)+randi([-30 30]);
    im(:,:,3) = im(:,:,3)+randi([-30 30]);
end
newIm = im;