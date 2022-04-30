% Run setup.m before running this script.
symVars = [l0, l1, l2, r0, r1, r2, m1, m2 ,m3, Ix1, Iy1, Iz1, Ix2, Iy2, Iz2, Ix3, Iy3, Iz3, g];

%Length of each link is 1m, distance of com is at centroid
%Link is cylindrical in shape with mass as 1 kg 
%radius of link is 0.1meters
%Inertia tensor is 0.0858, 0.0858, 0.005

values = [1, 1, 1, 0.5, 0.5, 0.5, 1, 1, 1, 0.0858 , 0.0858, 0.005, 0.0858, 0.005, 0.0858, 0.0858, 0.005, 0.0858, -10];
M_theta = subs(M_theta, symVars, values);
M(theta_1, theta_2, theta_3) = M_theta;

N_theta = subs(N_theta, symVars, values);
N(theta_1, theta_2, theta_3) = N_theta;

C_theta = subs(C_ijk, symVars, values);
syms theta1dot theta2dot theta3dot
temp = [theta1dot theta2dot theta3dot];
newC = sym(zeros(3,3));
for i = 1:3
    for j=1:3
        for k=1:3
            newC(i,j) = newC(i,j) + C_theta(i,j,k).*temp(k);
        end
    end
end
C(theta_1, theta_2, theta_3, theta1dot, theta2dot, theta3dot) = newC;

%% Initialize the control variables
thetaD = [0; -pi/2;0];
thetaD_dot = [0; 0; 0];
thetaD_ddot = [0 ; 0; 0];
Kv = 7;
Kp = 12.5;

thetas = [];
thetaDotss = [];
thetaDDotss = [];
torques= [];

for kd = 1:0.1:100
    theta = (2*pi)*randn(3,1) + (-pi);
    thetaDot = (2)*randn(3,1) + (-1);
    thetaDDot = (2)*randn(3,1) + (-1);
    h = 0.01;
    fprintf("Starting Control Loop \n");
    disp(kd);
    while ( sum(abs(thetaD - theta)) > 1e-4 )
    %     fprintf(".");
        %Store the states for plotting later
        thetas = [thetas, theta];
        thetaDotss = [thetaDotss, thetaDot];
    
        %Control inputs from PD controller.
        inputTorques = eval(N(theta(1), theta(2), theta(3)) - Kv.*(thetaDot) -Kp.*(theta - thetaD));
        torques = [torques, inputTorques];
        %Applying forward euler to estimate states.
        M_curr = eval(M(theta(1), theta(2), theta(3)));
        C_curr = eval(C(theta(1), theta(2), theta(3), thetaDot(1), thetaDot(2), thetaDot(3)));
        N_curr = eval(N(theta(1), theta(2), theta(3)));
        
        f1 = inv(M_curr)*(inputTorques - C_curr*thetaDot - N_curr);
        thetaDDotss = [thetaDDotss f1];
       
        newThetaDot = thetaDot + h.*f1;
        newTheta = theta + h.*thetaDot;
        
        theta = newTheta;
        thetaDot = newThetaDot;
    end
end

filename='ID_Data';
save(filename,'thetas','thetaDotss','thetaDDotss','torques')
