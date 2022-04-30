%filtering; to filter out high input values for training purposes
xt = load('ID_Data.mat');

the = sum(xt.thetas);
TF1 = find(isnan(the));
xt.thetas(:,TF1) = []; xt.thetaDotss(:,TF1) = []; xt.thetaDDotss(:,TF1) = []; xt.torques(:,TF1) = [];

tdot = sum(xt.thetaDotss);
TF2  = find(isnan(tdot));
xt.thetas(:,TF2) = []; xt.thetaDotss(:,TF2) = []; xt.thetaDDotss(:,TF2) = []; xt.torques(:,TF2) = [];

tDdot = sum(xt.thetaDDotss);
TF3   = find(isnan(tDdot)); 
xt.thetas(:,TF3) = []; xt.thetaDotss(:,TF3) = []; xt.thetaDDotss(:,TF3) = []; xt.torques(:,TF3) = [];

torqe = sum(xt.torques);
TF4   = find(abs(torqe)> 10); 
xt.thetas(:,TF4) = []; xt.thetaDotss(:,TF4) = []; xt.thetaDDotss(:,TF4) = []; xt.torques(:,TF4) = [];

thetas = xt.thetas;
thetaDotss = xt.thetaDotss;
thetaDDotss = xt.thetaDDotss;
torques = xt.torques;
filename='ID_Data';
save(filename,'thetas','thetaDotss','thetaDDotss','torques')