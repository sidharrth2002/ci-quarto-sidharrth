# A Hybrid Quarto Agent

For tournament, please use FinalPlayer with the thresholds specified in `main.py`.

Sidharrth Nagappan, 2023

### Player

<img src="./methodology.drawio.png" alt="methodology" style="zoom:50%;" />

### Repository Structure

- `DeepQNetwork` generates a deep Q-network that can be trained to play Quarto
- `GeneticAlgorithm` uses Genetic Algorithm to decide score thresholds for the hybrid player
- `MCTS2` contains the final version of the Monte Carlo Tree Search algorithm
- `MCTS` contains the first version of the Monte Carlo Tree Search algorithm
- `lib` contains my utility functions, such as board transformation and scoring functions
- `misc` for junk
- `QLMCTS` contains a QL-agent backed by MCTS
- `quarto` contains the Quarto game logic (Calabrese's class, my child class and custom OpenAI Gym environment)
- `report` contains the report for this project

To professor Calabrese, if there are any clarifications required about folder structure, I'll be happy to answer on Telegram, email or Github Issues, thanks!
