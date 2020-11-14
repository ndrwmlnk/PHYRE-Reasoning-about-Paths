# Solving Physics Puzzles by Reasoning about Paths

[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE)

**PHYRE** is a benchmark for physical reasoning.

![phyre](imgs/phyre_tasks.gif)

# Abstract

We propose a new deep learning model for goal-driven tasks that require intuitive physical reasoning and intervention in the scene to achieve a desired end goal. Its modular structure is motivated by hypothesizing a sequence of intuitive steps that humans apply when trying to solve such a task. The model first predicts the path the target object would follow without intervention and the path the target object should follow in order to solve the task. Next, it predicts the desired path of the action object and generates the placement of the action object. All components of the model are trained jointly in a supervised way; each component receives its own learning signal but learning signals are also backpropagated through the entire architecture. To evaluate the model we use PHYRE - a benchmark test for goal-driven physical reasoning in 2D mechanics puzzles.

# Action generation model

<p align="center"><img width="100%" src="imgs/model_pipeline.png" /></p>

**Top:** Action generation pipeline. NNs modules are highlighted with green rectangles. The task's initial scene is presented to the agent as five bitmap channels; one channel for each object class: Green target-object, blue dynamic goal-object, blue static goal-object, dynamic grey objects, static black objects.  **Bottom left** Model prediction examples. All examples of the generated *final action* in the figure solve the corresponding tasks.  **Bottom right:** Model architecture details: Every Conv2d and ConvTransposed2d Layer has a kernel size of 4x4, stride of 2 and padding of 1.

# Related projects

[PHYRE](https://phyre.ai) is a Benchmark For Physical Reasoning

# Citation

```bibtex
@article{harter2020phyre,
    title={Solving Physics Puzzles by Reasoning about Paths},
    author={Augustin Harter and Andrew Melnik and Gaurav Kumar and Dhruv Agarwal and Animesh Garg and Helge Ritter},
    year={2020},
    journal={arXiv:2020.00000}
}
```
