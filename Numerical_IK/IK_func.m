function [mse, percentage_converged] = IK_func(max_steps, num_points)

arguments
    max_steps(1, 1) = 10000000
    num_points(1, 1) = 20
end

load('..\data_3dof\IKQ_Data.mat', 'angles', 'configuration');
angles = wrapToPi(angles);

xi1 = [0, 0, 0, 0, 0, 1]';
xi2 = [0, -1, 0, -1, 0, 0]';
xi3 = [0, -1, 1, -1, 0, 0]';
xi = [xi1, xi2, xi3]; % 6xN

g0 = [eye(3), [0; 2; 1]; 0, 0, 0, 1];

%% Calculate the RMSE on random points points

index = randperm(90000, num_points);

configuration = configuration(:, index)';
angles = angles(:, index)';

num_points = size(angles, 1);

mse = zeros(num_points, 1);

num_converged = 0;

parfor (i = 1:size(angles, 1), 28)
    configuration_i = configuration(i, :);
    x_i = [configuration_i(1:3), configuration_i(5:7), configuration_i(4)];
    theta_s = zeros(3, 1);
    x_s = FK(xi, theta_s, g0);
    [theta_synthetic, converged] = IK_Pseudoinverse(xi, theta_s, x_s, x_i, g0, 0.001, 0.01, max_steps);
    error = mod(theta_synthetic'-angles(i, :)+pi, 2*pi) - pi;

    mse(i) = mean(error.^2);
    num_converged = num_converged + converged;
end

percentage_converged = num_converged / num_points;
end