function J = buildJacobianforLinks( xi_1, xi_2, xi_3 ,theta_1, theta_2, theta_3, gsl_0, num_of_links)
    if(num_of_links == 1)
        
        A1 = expm(getTwistMatrix(xi_1).*theta_1) * gsl_0;
        xi_1Phat = getAdjointInverse(A1, xi_1);
        xi_1P = getTwistVectorFromMatrix(xi_1Phat);

        J = [xi_1P, zeros(6,1), zeros(6,1)];
    end
    if(num_of_links == 2)

            A1 = expm(getTwistMatrix(xi_1).*theta_1) * expm(getTwistMatrix(xi_2).*theta_2) * gsl_0;
            xi_1Phat = getAdjointInverse(A1, xi_1);
            xi_1P = getTwistVectorFromMatrix(xi_1Phat);

            A2 = expm(getTwistMatrix(xi_2).*theta_2) * gsl_0;
            xi_2Phat = getAdjointInverse(A2, xi_2);
            xi_2P = getTwistVectorFromMatrix(xi_2Phat);
            J = [xi_1P, xi_2P, zeros(6,1)];
    end  
    if(num_of_links ==3)

            A1 = expm(getTwistMatrix(xi_1).*theta_1) * expm(getTwistMatrix(xi_2).*theta_2) * expm(getTwistMatrix(xi_3).*theta_3)* gsl_0;
            xi_1Phat = getAdjointInverse(A1, xi_1);
            xi_1P = getTwistVectorFromMatrix(xi_1Phat);

            A2 = expm(getTwistMatrix(xi_2).*theta_2) * expm(getTwistMatrix(xi_3).*theta_3) * gsl_0;
            xi_2Phat = getAdjointInverse(A2, xi_2);
            xi_2P = getTwistVectorFromMatrix(xi_2Phat);

            A3 = expm(getTwistMatrix(xi_3).*theta_3) * gsl_0;
            xi_3Phat = getAdjointInverse(A3, xi_3);
            xi_3P = getTwistVectorFromMatrix(xi_3Phat);
            
            J = [xi_1P, xi_2P, xi_3P];
    end
end