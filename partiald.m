%function to find the partial derivative
%calculates the partial derivative of the normailzed line integral
%holding the centre coordinates constant
%and then smooths it by a gaussian of appropriate sigma 
%%rmin and rmax are the minimum and maximum values of radii expected
%function also returns the maximum value of blur and the corresponding radius
%It also returns the finite differnce vector blur
%INPUTS:
%I;input image
%C:centre coordinates
%rmin,rmax:minimum and maximum radius values
%n:number of sides of the polygon(for lineint)
%part:specifies whether it is searching for the iris or pupil
%sigma:standard deviation of the gaussian
%OUTPUTS:
%blur:the finite differences vector
%r:radius at maximum value of 'blur'
%b:maximum value of 'blur'
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
function [b,r,blur]=partiald(I,C,rmin,rmax,sigma,n,part);
R=rmin:rmax;
count=size(R,2);
for k=1:count
[L(k)]=lineint(I,C,R(k),n,part);%computing the normalized line integral for each radius
if L(k)==0%if L(k)==0(this case occurs iff the radius takes the circle out of the image)
    %In this case,L is deleted as shown below and no more radii are taken for computation
    %(for that particular centre point).This is accomplished using the break statement
     L(k)=[];
    break;
end
end
D=diff(L);
D=[0 D];
%append one element at the beginning to make it an n vector
%Partial derivative at rmin is assumed to be zero
if strcmp(sigma,'inf')==1%the limiting case of the gaussian with sigma infinity(pls remember to change the code)strcmp syntax is different
f=ones(1,7)/7;
else
f=fspecial('gaussian',[1,5],sigma);%generates a 5 member 1-D gaussian
end
blur=convn(D,f,'same');%Smooths the D vecor by 1-D convolution 
%'same' indicates that size(blur) equals size(D)
blur=abs(blur);
[b,i]=max(blur);
r=R(i);
b=blur(i);
%calculates the blurred partial derivative
