clc ,clear all, close all

M1 = load('cost1.txt') ;
M2 = load('cost2.txt') ;
M3 = load('cost3.txt') ;
M4 = load('cost4.txt') ;

figure;
plot(M1);
title('Grad');
figure;
plot(M2);
title('Momentum');
figure;
plot(M3);
title('RMSprop');
figure;
plot(M4);
title('Adma');

figure;
plot(M1);
hold on;

plot(M2);
hold on;

plot(M3);
hold on;

plot(M4);
hold on;

legend('Grad','Momentum','RMSprop','Adma') ;