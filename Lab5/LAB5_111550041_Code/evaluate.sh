#!/bin/bash

# Task 1: CartPole Vanilla DQN
echo "Evaluating Task 1 (CartPole)..."
python ./LAB5_111550041_Code/test_model_task1.py --model-path LAB5_111550041_task1.pt --episodes 20

# Task 2: Pong Vanilla DQN
echo "Evaluating Task 2 (Pong Vanilla)..."
python ./LAB5_111550041_Code/test_model.py --model-path LAB5_111550041_task2.pt --episodes 20

# Task 3: Pong Enhanced DQN (Snapshots)
echo "Evaluating Task 3 (Pong Enhanced - 2.5M steps)..."
python ./LAB5_111550041_Code/test_model.py --model-path LAB5_111550041_task3_2500000.pt --episodes 20

# Task 3: Best Model
echo "Evaluating Task 3 (Best Model)..."
python ./LAB5_111550041_Code/test_model.py --model-path LAB5_111550041_task3_best.pt --episodes 20