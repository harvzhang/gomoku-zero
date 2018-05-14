
MCTS Memory Buffer
------------------


This implementation incorporates the element of memory in the MCTS which is inspired by

Monte-Carlo Tree Search and Rapid Action Value Estimation in Computer Go [http://www.cs.utexas.edu/~pstone/Courses/394Rspring13/resources/mcrave.pdf]

To turn on MCTS memory buffer, in 'normal.py', set 'self.mc_rave = True'


Increasing Residual Net Layers
------------------------------

To automatically increase the residual network layers after reaching a certain consecutive loss count,

in 'normal.py', set 'self.increase_res_layers = True'

To adjust the number of consectuive losses before update, set 'self.loss_before_update'

Src File Structure
------------------

The implementation file structure is divided as follows:

/agent:
- api_gomoku.py: predict API for the model
- model_gomoku.py: builds the model architecture and handles saving/loading model
- player_gomoku.py: computes player action by performing MCTS

/configs:
- normal.py: normal training and play configurations (recommended)
- mini.py: reduced set of configurations

/env:
- gomoku_env.py: defines the environment for the Gomoku

/lib:
- data_helper.py: save/load game data and fetch candidate model paths
- logger.py: performs logging
- model_helper.py: save/load models
- tf_util.py: manage tensorflow session

/play_game:
- game_model.py: manage game between human play and gomoku zero
- gui.py: display the game interface

/worker:
- evaluate.py: evaluates the model based on
- optimize.py: trains the model architecture using generated play data
- self_play.py: use current best model to perform self-play

manager.py: starts appropriate thread based on specified process

config.py: manages all configuration settings

run.py: entry point of program
