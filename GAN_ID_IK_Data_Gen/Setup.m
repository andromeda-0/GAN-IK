% Estimates the EOMs of a 3link manipulator
clear all;
clc;

syms l0 l1 l2 r0 r1 r2
syms theta_1 theta_2 theta_3

%Twist parameters
w_1 = [ 0;  0;  1];
q_1 = [ 0;  0; l0];
w_2 = [-1;  0;  0];
q_2 = [ 0;  0; l0];
w_3 = [-1;  0;  0];
q_3 = [ 0; l1; l0];

%Twist Vectors
xi_1 =[-cross(w_1, q_1); w_1];
xi_2 =[-cross(w_2, q_2); w_2];
xi_3 =[-cross(w_3, q_3); w_3];

%Get Jacobians for each link
gsl1_0 = [eye(3,3),  [0;0;r0]; 0, 0, 0, 1];
gsl2_0 = [eye(3,3), [0;r1;l0]; 0, 0, 0, 1];
gsl3_0 = [eye(3,3), [0;l1+r2;l0]; 0, 0, 0, 1];

J_1 = simplify(buildJacobianforLinks(xi_1, xi_2, xi_3, theta_1, theta_2, theta_3, gsl1_0 , 1));
J_2 = simplify(buildJacobianforLinks(xi_1, xi_2, xi_3, theta_1, theta_2, theta_3, gsl2_0 , 2));
J_3 = simplify(buildJacobianforLinks(xi_1, xi_2, xi_3, theta_1, theta_2, theta_3, gsl3_0 , 3), 'Steps', 30);

%Define the Generalised Inertia matrix
syms m1 m2 m3 Ix1 Iy1 Iz1 Ix2 Iy2 Iz2 Ix3 Iy3 Iz3
genM1 = [eye(3,3).*m1, zeros(3,3); zeros(3,3), eye(3,3).*[Ix1; Iy1; Iz1]];
genM2 = [eye(3,3).*m2, zeros(3,3); zeros(3,3), eye(3,3).*[Ix2; Iy2; Iz2]];
genM3 = [eye(3,3).*m3, zeros(3,3); zeros(3,3), eye(3,3).*[Ix3; Iy3; Iz3]];


%% Estimate the Inertia matrix of the System, M(theta) = Sum(J^T * M * J)

M_theta = (transpose(J_1) * genM1 * J_1) + (transpose(J_2) * genM2 * J_2) + (transpose(J_3) * genM3 * J_3);
M_theta = rewrite(M_theta, 'sincos');
M_theta = simplify(M_theta);

fprintf("\n The Inertia Matrix of the system is given by M(theta) \n")
for i =1:3
    for j = 1:3 
        fprintf('M(%d,%d) = ', i, j);
        disp(M_theta(i,j))
    end
end
%% Estimate the Coriolis  and Centrifugal Forces, C(theta, theta_dot)
thetas = [theta_1, theta_2, theta_3];
C_ijk = sym(zeros(3,3,3));
for i = 1:3
    for j = 1:3
        for k = 1:3
            term1 = diff(M_theta(i,j), thetas(k));
            term2 = diff(M_theta(i,k), thetas(j));
            term3 = diff(M_theta(k,j), thetas(i));
            final_t = sym(0.5) *(term1 + term2 - term3);
            C_ijk(i,j,k) = combine(rewrite(final_t, 'sincos'));
        end
    end
end

fprintf("\n The Coriolis Matrix can be written as: (C(i,j,k), (k is the joint_velocity term)) \n")
for i = 1:3
    for j = 1:3
        for k = 1:3
            fprintf(" C(%d, %d, %d) = ", i, j, k);
            disp(C_ijk(i,j,k))
        end
    end
end

%% Compute the gravitational terms

syms g %gravity

%Get the heights based on configuration using forward kinematics map
gsl1T =  getForwardKinematicsMap(xi_1, xi_2, xi_3, theta_1, theta_2, theta_3, gsl1_0, 1);
gsl2T =  getForwardKinematicsMap(xi_1, xi_2, xi_3, theta_1, theta_2, theta_3, gsl2_0, 2);
gsl3T =  getForwardKinematicsMap(xi_1, xi_2, xi_3, theta_1, theta_2, theta_3, gsl3_0, 3);

%Height is the z-term (g_34) of the homogenous transform
h1_theta = gsl1T(3,4);
h2_theta = simplify(rewrite(gsl2T(3,4), 'sincos'), 'Steps', 10);
h3_theta = simplify(rewrite(gsl3T(3,4), 'sincos'), 'Steps', 10);

%Potential Energy is defined as
V_theta = (m1*g*h1_theta) + (m2*g*h2_theta) + (m3*g*h3_theta);

N_theta = sym(zeros(3,1));
for i = 1:3
    term = diff(V_theta, thetas(i));
    N_theta(i,1) = simplify(term, 'Steps', 15);
end

fprintf("The gravitational terms are N(theta, theta_dot) are \n")
for i = 1:3
    fprintf(' N(%d, 1) = ', i);
    disp(N_theta(i))
end