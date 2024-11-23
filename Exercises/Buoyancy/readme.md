Buoyancy Simulation Exercise

This repository contains a simple simulation of the buoyancy of an object in a fluid with a different density. When the object is released, it moves to reach equilibrium based on its density and the fluid's density. There are two scenarios:

No friction: The object oscillates permanently without damping, maintaining a constant amplitude.
With friction: A friction parameter is introduced, causing the object to gradually approach its equilibrium position.

Files

The repository contains three files for the exercise:

buoyant.f95: Fortran code that calculates the vertical movement of the object based on its density, the fluid density, and the buoyant force.
buoyant.sce: Scilab code that generates an animation of the object’s movement, showing how it oscillates or approaches equilibrium depending on the friction parameter.
buoyant.py: Python code that combines both the calculations and the animation in one file, offering a more streamlined approach.

How It Works

Fortran code (buoyant.f95): The Fortran code simulates the vertical movement of a buoyant object. The object's movement is governed by the balance between the buoyant force and the gravitational force. In the frictionless scenario, the object oscillates indefinitely. In the scenario with friction, the friction parameter damps the oscillations, gradually bringing the object to its equilibrium position.

Scilab code (buoyant.sce): The Scilab code reads the output of the Fortran simulation and visualizes it through an animation. The code shows the object's position over time, with color changes indicating whether the object is too light or too heavy compared to its equilibrium position.

Python code (buoyant.py): The Python version combines both the simulation and the animation into a single script. The calculations are done using Python and numpy, and the animation is generated using matplotlib.

Customization

Users are encouraged to modify the following parameters to observe how different conditions affect the buoyancy simulation:

Object density: Change the density of the object to see how it affects the equilibrium position and oscillation behavior.
Fluid density: Modify the density of the fluid to explore how the buoyant force changes with different mediums.
Friction parameter: Adjust the friction coefficient to see how it affects the damping of the oscillations and the rate at which the object reaches equilibrium.

Requirements

To run the Python code (buoyant.py), make sure you have the following dependencies installed:

numpy for numerical calculations.
matplotlib for animation and plotting.

To install these dependencies, run:

pip install numpy matplotlib

Running the Code

For Python: Simply run the buoyant.py file using Python.

    python buoyant.py

For Fortran and Scilab:
    Compile and run the Fortran code (buoyant.f95).
    Run the Scilab code (buoyant.sce) to visualize the output.

License

This project is licensed under the MIT License - see the LICENSE file for details.