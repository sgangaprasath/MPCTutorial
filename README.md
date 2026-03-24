# MPCTutorial

A tutorial on Model Predictive Control (MPC) using a toy problem of rocket landing. This repository has the LaTeX files and the implementation of MPC for controlling a rocket's descent to a landing pad.

## Project Structure

### CasADi/
This folder contains the MPC implementation using the CasADi framework.

- `RocketMPC.py`: The main Python script implementing the MPC controller for the rocket landing problem. It defines the dynamics, constraints, and optimization problem using CasADi.

## Description

The MPC code solves a simplified rocket landing problem where the goal is to control the rocket's thrust to achieve a soft landing on a target pad while respecting constraints on fuel and dynamics. The LaTeX report provides the detailed mathematical derivation of the model, constraints, and optimization formulation.

### LaTeX/
This folder contains the LaTeX report detailing the mathematical calculations and theory behind the MPC implementation.