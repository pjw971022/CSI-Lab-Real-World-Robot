# Sembot

#### [[Video]](https://[www.youtube.com/watch?v=example](https://youtu.be/TSAJOjf2C5E?si=wkk0GVENzLFQZ1fL)

Sembot is a project developed by CSI Lab for creating real-world demonstration videos. This method leverages large language models and vision-language models for zero-shot trajectory synthesis in manipulation tasks. This repository provides the implementation of Sembot in RLBench and Real-world.

<img src="media/sembot_teaser.gif" width="550">

Sembot is a method that leverages large language models and vision-language models for zero-shot trajectory synthesis in manipulation tasks. This repository provides the implementation of Sembot in [RLBench](https://sites.google.com/view/rlbench).

## Key Features

- Zero-shot method: No training data required
- Uses Language Model Programs (LMPs) to decompose instructions and compose value maps
- Implements a greedy planner for trajectory planning
- Provides controllers for robot actions and environment dynamics modeling

## Quick Start

1. Set up the environment (see [Setup Instructions](#setup-instructions))
2. Obtain an [OpenAI API](https://openai.com/blog/openai-api) key
3. Run the demo in `src/playground.ipynb`

## Main Components

- `playground.ipynb`: Main demo notebook
- `LMP.py`: Implementation of Language Model Programs
- `interfaces.py`: APIs for language models to operate in voxel space
- `planners.py`: Greedy planner for trajectory planning
- `controllers.py`: Robot action controllers
- `dynamics_models.py`: Environment dynamics model

For a complete list of components and their descriptions, see [Code Structure](#code-structure).

## Setup Instructions
For detailed setup instructions, please refer to [VoxPoser GitHub Repository](https://github.com/huangwl18/VoxPoser).

## Acknowledgments

This project builds upon several existing works:

- Skill decoder framework and overall approach: [VoxPoser](https://github.com/huangwl18/VoxPoser)
- Environment: [RLBench](https://sites.google.com/view/rlbench)

- LMP implementation: [Code as Policies](https://code-as-policies.github.io/)

- Some code snippets: [Where2Act](https://cs.stanford.edu/~kaichun/where2act/)

