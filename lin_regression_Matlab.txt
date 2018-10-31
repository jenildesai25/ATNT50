
% The following is a MATLAB code for Linear Regresion

% Following transform Xtrain and Xtest into double precision if they are not.

% Xtrain = double(Xtrain);
%  Xtest = double(Xtest);


B = pinv(Xtrain') *  double(Ytrain)'  ; % (XX')^{-1} X  * Y'
Ytrain1 = B' * Xtrain;
Ytest1 = B' * Xtest;

[Ytest2value  Ytest2]= max(Ytest1,[],1);
[Ytrain2value  Ytrain2]= max(Ytrain1,[],1);
