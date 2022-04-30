# GAN-IK
Generative Adversarial Networks for Inverse Kinematics & Inverse Dynamics.

Course Project for 16-711 Kinematics, Dynamics, and Control. Spring 2022.

Authors: Zongyue Zhao, Nishant Mohanty, Akshay Dharmavaram

----
For the main results we presented in our report, the generator contained five linear layers with 256 neurons each and an output linear layer with $\mathbf{dim}(O)$ neurons. The activation function between hidden layers was chosen to be LeakReLU with a 0.1 slope. No activation was used after the output layer. The discriminator contained two hidden linear layers with 256 units and an output linear layer with a single neuron. The intermediate activation functions were the same as the generator, and a sigmoid function was applied after the discriminator's output layer. All models with trained with an Adam optimizer with a learning rate of 2e-4.
