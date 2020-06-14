# Salamander-Robotica Simulation
##### Florian Genilloud, Emilio Fernández, Joachim Durant

This is the final project for Computational Motor Control (2020 EPFL course). We made use of the Central Pattern Generator model explained in class for simulating a salamander capable of both swimming and walking. What's even more important, we were sucessful in developing a smooth transition from walking to swimming thanks to a fine parameter tuning. We explain the totality of our results in this [report](report.pdf).

![](simulation_final.gif)

The project is divided in several subexercises where we increase the model complexity step by step: 
+ First, we set all the parameters in `robot_parameters.py` and define the differential equations in `network.py` so that we can run  `exercise_example.py`without a problem and the behaviour is similar to the expected one in theory.
+ Next, in `exercise_8b.py`, we run a first grid search in order to obtain the most efficient (high speed and low energy) values for the amplitude gradient and the phase lag when swimming.
+ In `exercise_8c.py`, we study more in detail how does the amplitude gradient influence the speed and energy of the salamander.
+ In `exercise_8d.py`, we experiment again with the paramaters to induce turning and backwards swimming
+ In `exercise_8f.py`, we find the most optimal parameter values for a correct coordination between spine and limb joints
+ Finally, in `exercise_8d.py`, we adapt the salamander drive depending on its position in order to obtain a smooth land to water transition.

