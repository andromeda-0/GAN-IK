%run the script after running setup.m and ID_Data.m
%generate ik kinematics data
w_1 = [ 0;  0;  1];
q_1 = [ 0;  0;  1];
w_2 = [-1;  0;  0];
q_2 = [ 0;  0;  1];
w_3 = [-1;  0;  0];
q_3 = [ 0;  1;  1];

%Twist Vectors
xi_1 =[-cross(w_1, q_1); w_1];
xi_2 =[-cross(w_2, q_2); w_2];
xi_3 =[-cross(w_3, q_3); w_3];

%Zero-transform for each link
gsl1_0 = [eye(3,3), [0; 0; 1]; 0, 0, 0, 1];
gsl2_0 = [eye(3,3), [0; 1; 1]; 0, 0, 0, 1]; 
gsl3_0 = [eye(3,3), [0; 2; 1]; 0, 0, 0, 1];

xt = load('ID_Data.mat');
thetas = xt.thetas;
angles = thetas;

configuration = [];

for i = 1:length(thetas)
%     disp(i)
    g1 = getForwardKinematicsMap(xi_1, xi_2, xi_3, thetas(1, i), thetas(2, i), thetas(3, i), gsl1_0, 1);
    g2 = getForwardKinematicsMap(xi_1, xi_2, xi_3, thetas(1, i), thetas(2, i), thetas(3, i), gsl2_0, 2);
    g3 = getForwardKinematicsMap(xi_1, xi_2, xi_3, thetas(1, i), thetas(2, i), thetas(3, i), gsl3_0, 3);
    
    x = [g1(1, 4), g2(1, 4), g3(1, 4)];
    y = [g1(2, 4), g2(2, 4), g3(2, 4)];
    z = [g1(3, 4), g2(3, 4), g3(3, 4)];

    configuration = [configuration [g3(1:3,4);rotm2quat(g3(1:3,1:3))']];
end

filename='IKQ_Data';
save(filename,'angles','configuration')
