#!/bin/bash

set -e

echo "========================================"
echo "Evaluating Task 1: A2C on Pendulum"
echo "========================================"
python a2c_pendulum.py --test --env-step 190000 --seed 700 --model-path ../LAB7_111550041_task1_a2c_pendulum.pt


echo -e "\n========================================"
echo "Evaluating Task 2: PPO on Pendulum"
echo "========================================"
python ppo_pendulum.py --test --env-step 30000 --seed 700 --model-path ../LAB7_111550041_task2_ppo_pendulum.pt


echo -e "\n========================================"
echo "Evaluating Task 3: PPO on Walker2d (1M steps)"
echo "========================================"
python ppo_walker.py --test --env-step 1000000 --seed 77 --model-path ../LAB7_111550041_task3_ppo_1m.pt


echo -e "\n========================================"
echo "Evaluating Task 3: PPO on Walker2d (1.5M steps)"
echo "========================================"
python ppo_walker.py --test --env-step 1500000 --seed 77 --model-path ../LAB7_111550041_task3_ppo_1p5m.pt


echo -e "\n========================================"
echo "Evaluating Task 3: PPO on Walker2d (2M steps)"
echo "========================================"
python ppo_walker.py --test --env-step 2000000 --seed 77 --model-path ../LAB7_111550041_task3_ppo_2m.pt


echo -e "\n========================================"
echo "Evaluating Task 3: PPO on Walker2d (2.5M steps)"
echo "========================================"
python ppo_walker.py --test --env-step 2500000 --seed 77 --model-path ../LAB7_111550041_task3_ppo_2p5m.pt


echo -e "\n========================================"
echo "Evaluating Task 3: PPO on Walker2d (3M steps)"
echo "========================================"
python ppo_walker.py --test --env-step 3000000 --seed 77 --model-path ../LAB7_111550041_task3_ppo_3m.pt


echo -e "\n========================================"
echo "Evaluating Task 3: PPO on Walker2d (Best Model)"
echo "========================================"
python ppo_walker.py --test --env-step -1 --seed 77 --model-path ../LAB7_111550041_task3_best.pt

echo -e "\nAll evaluations completed successfully!"