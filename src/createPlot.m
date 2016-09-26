X = [.0001 .0005 .001 .05 .1 .3 .5 1];
    
Y = [.7435 .7455 .7508 .7376 .7321 .7081 .6907 .6484];
plot(X,Y,'-ro');
title('Graph of alpha and corresponding accuracy');
xlabel('alpha');
ylabel('Accuracy Percentages');


axis([.0001, 1, .63, .76]);



% X = [95 86 80 76 61 46 36 26 16];
% 
% Y = [57 60.42 60.67 60.53 60 57.49 51.75 47.5 46];
% 
% plot(X,Y,'-ro');
% 
% title('Graph of depth of tree and corresponding accuracy');
% xlabel('Tree depth');
% ylabel('Accuracy Percentages');