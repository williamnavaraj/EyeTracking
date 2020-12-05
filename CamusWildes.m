%Original Viola Jones Hough transform based by Le Tan Phuc
%Modified to Camus&Wildes' method by William Navaraj
%William.Navaraj@ntu.ac.uk

%Ctrl + C to stop the code


clear all   
clf('reset');
objects = imaqfind %find video input objects in memory
delete(objects) %delete a video input object from memory

vidObj = videoinput('winvideo',1);
%I=getsnapshot(vid);
triggerconfig(vidObj, 'manual');
start(vidObj);

%cam=webcam();   %create webcam object

right=imread('RIGHT.jpg');
left=imread('LEFT.jpg');
noface=imread('no_face.jpg');
straight=imread('STRAIGHT.jpg');

detector = vision.CascadeObjectDetector(); % Create a detector for face using Viola-Jones
detector1 = vision.CascadeObjectDetector('EyePairSmall'); %create detector for eyepair

while true % Infinite loop to continuously detect the face
       vid = getsnapshot(vidObj);%snapshot(cam);
    %vid=snapshot(cam);  %get a snapshot of webcam
    vid = rgb2gray(vid);    %convert to grayscale
    img = flip(vid, 2); % Flips the image horizontally
    
     bbox = step(detector, img); % Creating bounding box using detector  
      
     if ~ isempty(bbox)  %if face exists 
         biggest_box=1;     
         for i=1:rank(bbox) %find the biggest face
             if bbox(i,3)>bbox(biggest_box,3)
                 biggest_box=i;
             end
         end
         faceImage = imcrop(img,bbox(biggest_box,:)); % extract the face from the image
         bboxeyes = step(detector1, faceImage); % locations of the eyepair using detector
         
         subplot(2,2,1),subimage(img); hold on; % Displays full image
         for i=1:size(bbox,1)    %draw all the regions that contain face
             rectangle('position', bbox(i, :), 'lineWidth', 2, 'edgeColor', 'y');
         end
         
         subplot(2,2,3),subimage(faceImage);     %display face image
                 
         if ~ isempty(bboxeyes)  %check it eyepair is available
             
             biggest_box_eyes=1;     
             for i=1:rank(bboxeyes) %find the biggest eyepair
                 if bboxeyes(i,3)>bboxeyes(biggest_box_eyes,3)
                     biggest_box_eyes=i;
                 end
             end
             
             bboxeyeshalf=[bboxeyes(biggest_box_eyes,1),bboxeyes(biggest_box_eyes,2),bboxeyes(biggest_box_eyes,3)/3,bboxeyes(biggest_box_eyes,4)];   %resize the eyepair width in half
             
             eyesImage = imcrop(faceImage,bboxeyeshalf(1,:));    %extract the half eyepair from the face image
             eyesImage = imadjust(eyesImage);    %adjust contrast

             r = bboxeyeshalf(1,4)/4;
             %[centers, radii, metric] = imfindcircles(eyesImage, [floor(r-r/4) floor(r+r/2)], 'ObjectPolarity','dark', 'Sensitivity', 0.965); % Hough Transform
             h=gcf;
             [a,b]=thresh(eyesImage, floor(r-r/4), floor(r+r/2));
             centers=[a(2), a(1)];
             radii=a(3);
             [M,I] = sort(radii, 'descend');
                 
             eyesPositions = centers;
               
             subplot(2,2,2),hold off,subimage(eyesImage); hold on;
              
             viscircles(centers, radii,'EdgeColor','b');
                  
             if ~isempty(centers)
                pupil_x=centers(1);
                disL=abs(0-pupil_x);    %distance from left edge to center point
                disR=abs(bboxeyeshalf(1,3)-pupil_x);%distance from right edge to center point
                subplot(2,2,4);
                if disR<bboxeyeshalf(1,3)/3
                    subimage(right);
                else if disL<bboxeyeshalf(1,3)/2
                    subimage(left);
                    else
                       subimage(straight); 
                    end
                end
     
             end          
         end
     else
        subplot(2,2,4);
        subimage(noface);
     end
     set(gca,'XtickLabel',[],'YtickLabel',[]);

   hold off;
   pause(0.1);
end

%function to search for the centre coordinates of the pupil and the iris
%along with their radii
%It makes use of Camus&Wildes' method to select the possible centre coordinates first
%The method consist of thresholding followed by
%checking if the selected points(by thresholding)
%correspond to a local minimum in their immediate(3*s) neighbourhood
%these points serve as the possible centre coordinates for the iris.
%Once the iris has been detected(using Daugman's method);the pupil's centre coordinates
%are found by searching a 10*10 neighbourhood around the iris centre and varying the radius
%until a maximum is found(using  Daugman's integrodifferential operator)
%INPUTS:
%I:image to be segmented
%rmin ,rmax:the minimum and maximum values of the iris radius
%OUTPUTS:
%cp:the parametrs[xc,yc,r] of the pupilary boundary
%ci:the parametrs[xc,yc,r] of the limbic boundary
%out:the segmented image
%Author:Anirudh S.K.
%Department of Computer Science and Engineering
%Indian Institute of Techology,Madras
% 
% -------------------------------------------------------------------------
% Copyright (c) 2007, Anirudh Sivaraman
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%     * Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the distribution
%       
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.
% -------------------------------------------------------------------------
function [ci,cp,out]=thresh(I,rmin,rmax);
scale=1;
%Libor Masek's idea that reduces complexity
%significantly by scaling down all images to a constant image size 
%to speed up the whole process
rmin=rmin*scale;
rmax=rmax*scale;
%scales all the parameters to the required scale
I=im2double(I);
%arithmetic operations are not defined on uint8
%hence the image is converted to double
pimage=I;
%stores the image for display
I=imresize(I,scale);
I=imcomplement(imfill(imcomplement(I),'holes'));
%this process removes specular reflections by using the morphological operation 'imfill'
%I=nbdavg(I);
%blurs the sharp image formed as a result of using imfill
rows=size(I,1);
cols=size(I,2);
[X,Y]=find(I<0.5);
%Generates a column vector of the image elements
%that have been selected by tresholding;one for x coordinate and one for y
s=size(X,1);
for k=1:s %
    if (X(k)>rmin)&(Y(k)>rmin)&(X(k)<=(rows-rmin))&(Y(k)<(cols-rmin))
            A=I((X(k)-1):(X(k)+1),(Y(k)-1):(Y(k)+1));
            M=min(min(A));
            %this process scans the neighbourhood of the selected pixel
            %to check if it is a local minimum
           if I(X(k),Y(k))~=M
              X(k)=NaN;
              Y(k)=NaN;
           end
    end
end
v=find(isnan(X));
X(v)=[];
Y(v)=[];
%deletes all pixels that are NOT local minima(that have been set to NaN)
index=find((X<=rmin)|(Y<=rmin)|(X>(rows-rmin))|(Y>(cols-rmin)));
X(index)=[];
Y(index)=[];  
%This process deletes all pixels that are so close to the border 
%that they could not possibly be the centre coordinates.
N=size(X,1);
%recompute the size after deleting unnecessary elements
maxb=zeros(rows,cols);
maxrad=zeros(rows,cols);
%defines two arrays maxb and maxrad to store the maximum value of blur
%for each of the selected centre points and the corresponding radius
for j=1:N
    [b,r,blur]=partiald(I,[X(j),Y(j)],rmin,rmax,'inf',600,'iris');%coarse search
    maxb(X(j),Y(j))=b;
    maxrad(X(j),Y(j))=r;
end
[x,y]=find(maxb==max(max(maxb)));
ci=search(I,rmin,rmax,x,y,'iris');%fine search
%finds the maximum value of blur by scanning all the centre coordinates
ci=ci/scale;
%the function search searches for the centre of the pupil and its radius
%by scanning a 10*10 window around the iris centre for establishing 
%the pupil's centre and hence its radius
cp=search(I,round(0.1*r),round(0.8*r),ci(1)*scale,ci(2)*scale,'pupil');%Ref:Daugman's paper that sets biological limits on the relative sizes of the iris and pupil
cp=cp/scale;
%displaying the segmented image
%out=drawcircle(pimage,[ci(1),ci(2)],ci(3),600);
%out=drawcircle(out,[cp(1),cp(2)],cp(3),600);
end


