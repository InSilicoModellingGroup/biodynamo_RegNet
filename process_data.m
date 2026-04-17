
clear, clc, close all

if false
    data = readmatrix('sine.csv');

    figure, hold on
    plot(data(:,1),data(:,2),'r-')
    plot(data(1,1),data(1,2),'r*')
    axis equal
    hold off
end

if false
    data = readmatrix('oscillator.csv');
    
    figure, hold on
    plot(data(:,2),data(:,3),'r-')
    plot(data(1,2),data(1,3),'r*')
    axis equal
    hold off
end

if false
    data = readmatrix('lorenz.csv');
    
    figure, hold on
    plot3(data(:,2),data(:,3),data(:,4),'r-')
    plot3(data(1,2),data(1,3),data(1,4),'r*')
    axis equal
    hold off
end