import os
from logging import getLogger
from random import random
from time import sleep
import numpy as np

from gomoku_zero.agent.model_gomoku import GomokuModel
from gomoku_zero.agent.player_gomoku import GomokuPlayer
from gomoku_zero.config import Config
from gomoku_zero.env.gomoku_env import GomokuEnv, Winner, Player
from gomoku_zero.lib import tf_util
from gomoku_zero.lib.data_helper import get_next_generation_model_dirs
from gomoku_zero.lib.model_helpler import save_as_best_model, load_best_model_weight, load_baseline_model_weight

logger = getLogger(__name__)


def start(config: Config, res_add=False, mcrave=False):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.2)
    return EvaluateWorker(config, res_add, mcrave).start()


class EvaluateWorker:
    def __init__(self, config: Config):
        '''
        Evaluates the candidate models against the current best model
        Also evaluates the best model against the baseline model
        :param config: configuration settings
        '''
        self.config = config
        self.best_model = None
        self.baseline_model = None

        # record the game data
        self.win_counter = self.config.eval.record_num
        self.generation_counter = self.config.eval.evaluate_generation
        self.model_win_rate = []
        self.num_res_layers = []
        self.game_result = []
        self.is_white = []

        self.turn = 0

        # create the directories to save file
        if not os.path.exists(self.config.resource.plot_dir):
            os.makedirs(self.config.resource.plot_dir)

        # load from existing
        win_rate_file = os.path.join(self.config.resource.plot_dir, self.config.resource.winning_rate_filename)
        if os.path.isfile(win_rate_file):
            print('Loading Win Rate File From Previous Saved Data')
            self.model_win_rate = np.load(win_rate_file).tolist()

    def start(self):
        ''' the evaluator thread begins here '''

        # load the current best model if exists
        self.best_model = self.load_best_model()

        # load the baseline model
        self.baseline_model = self.load_baseline_model()

        while True:
            # load the next generation model and evaluate against
            ng_model, model_dir = self.load_next_generation_model()
            logger.debug(f"start evaluate model {model_dir}")
            ng_is_great = self.evaluate_model(ng_model)
            if ng_is_great:
                logger.debug(f"New Model become best model: {model_dir}")
                save_as_best_model(ng_model)
                self.best_model = ng_model
            self.remove_model(model_dir)


            # record the number of consecutive wins from the model
            self.record_win(ng_is_great)
            logger.debug(f"Should Increase Res Layer %r" % self.should_add_layers())

            # update the number of res layers if achieve consecutive losses
            if self.config.trainer.increase_res_layers and self.should_add_layers():
                ng_model = Connect4Model(self.config)
                res_layers = self.best_model.get_num_res_layers()
                res_layers = min(res_layers + 1, self.config.model.max_res_layer_num)
                ng_model.build(res_layers)
                ng_model.update_model(self.best_model)
                logger.debug(f"New Model (With Added Layers) become best model: {model_dir}")
                logger.debug(f"New Model has %d res layers" % res_layers)
                save_as_best_model(ng_model)
                self.best_model = ng_model

            # count the number of model generations until baseline test
            self.generation_counter = max(self.generation_counter - 1, 0)
            self.turn += 1
            logger.debug(f"Turn %d, Generation Counter %d " % (self.turn, self.generation_counter))
            if self.generation_counter == 0:
                logger.debug(f"Evaluating Model Against Baseline")
                self.evaluate_against_baseline()
                self.generation_counter = self.config.eval.evaluate_generation

    def record_win(self, ng_is_great):
        ''' Records the number of consecutive best model wins until 'loss_before_update' '''
        if ng_is_great:
            self.win_counter = self.config.trainer.loss_before_update
        else:
            self.win_counter = max(self.win_counter - 1, 0)

    def should_add_layers(self):
        ''' Determine if res layer should be added to the existing model'''
        return self.win_counter == 0

    def evaluate_model(self, ng_model):
        '''
        Evaluates the candidate model against the current best model
        :param ng_model: candidate model
        :return: True if candidate model achieves win rate better than replacement rate
        '''
        results = []
        winning_rate = 0
        for game_idx in range(self.config.eval.game_num):
            # ng_win := if ng_model win -> 1, lose -> 0, draw -> None
            ng_win, white_is_best = self.play_game(self.best_model, ng_model)
            if ng_win is not None:
                results.append(ng_win)
                winning_rate = sum(results) / len(results)
            logger.debug(f"game {game_idx}: ng_win={ng_win} white_is_best_model={white_is_best} "
                         f"winning rate {winning_rate*100:.1f}%")
            if results.count(0) >= self.config.eval.game_num * (1-self.config.eval.replace_rate):
                logger.debug(f"lose count reach {results.count(0)} so give up challenge")
                break
            if results.count(1) >= self.config.eval.game_num * self.config.eval.replace_rate:
                logger.debug(f"win count reach {results.count(1)} so change best model")
                break
        if len(results) != 0:
            winning_rate = sum(results) / len(results)
            logger.debug(f"winning rate {winning_rate*100:.1f}%")
        return winning_rate >= self.config.eval.replace_rate

    def evaluate_against_baseline(self):
        ''' Evaluates the current best model against the baseline model '''
        results = []
        winning_rate = 0
        games = []

        for game_idx in range(self.config.eval.game_baseline_num):
            # ng_win := if ng_model win -> 1, lose -> 0, draw -> None
            baseline_win, white_is_best = self.play_game(self.best_model, self.baseline_model)
            if baseline_win is not None:
                results.append(baseline_win)
                winning_rate = sum(results) / len(results)
            logger.debug(f"Against baseline - game {game_idx}: baseline_win={baseline_win} white_is_best_model={white_is_best} "
                    f"winning rate {winning_rate*100:.1f}%")

            # record the game outcome: win, tie, loss
            if baseline_win is None:
                games.append(0)
            elif baseline_win == 0:
                games.append(-1)
            else:
                games.append(1)

            # record color of the winner
            self.is_white.append(white_is_best)

        self.game_result.append(np.array(games))

        if len(results) != 0:
            winning_rate = sum(results) / len(results)
            logger.debug(f"Against baseline - baseline winning rate {winning_rate*100:.1f}%")
            self.model_win_rate.append(1.0 - winning_rate)
            self.num_res_layers.append(self.best_model.get_num_res_layers())

        self.save_num_res_layers()
        self.save_model_win_rate()

    def save_model_win_rate(self):
        ''' Save the win rate of the best model against baseline and the game data '''
        rc = self.config.resource
        np.save(os.path.join(rc.plot_dir, rc.winning_rate_filename), self.model_win_rate)
        np.save(os.path.join(rc.plot_dir, 'game_result.npy'), self.game_result)
        np.save(os.path.join(rc.plot_dir, 'is_white.npy'), self.is_white)

    def save_num_res_layers(self):
        rc = self.config.resource
        np.save(os.path.join(rc.plot_dir, rc.num_layers_filename), self.num_res_layers)

    def play_game(self, best_model, ng_model):
        ''' Plays a single game between the best model and candidate model'''

        env = GomokuEnv().reset()

        best_player = GomokuPlayer(self.config, best_model, play_config=self.config.eval.play_config)
        ng_player = GomokuPlayer(self.config, ng_model, play_config=self.config.eval.play_config)
        best_is_white = random() < 0.5
        if not best_is_white:
            black, white = best_player, ng_player
        else:
            black, white = ng_player, best_player

        env.reset()
        while not env.done:
            if env.player_turn() == Player.black:
                action = black.action(env.board, env.turn)
            else:
                action = white.action(env.board, env.turn)
            env.step(action)

        # record the winner
        ng_win = None
        if env.winner == Winner.white:
            if best_is_white:
                ng_win = 0
            else:
                ng_win = 1
        elif env.winner == Winner.black:
            if best_is_white:
                ng_win = 1
            else:
                ng_win = 0
        return ng_win, best_is_white

    def load_best_model(self):
        model = GomokuModel(self.config)
        load_best_model_weight(model)
        return model

    def load_baseline_model(self):
        model = GomokuModel(self.config)
        load_baseline_model_weight(model)
        return model

    def load_next_generation_model(self):
        ''' Load the next generation model'''
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)
            if dirs:
                break
            logger.info(f"There is no next generation model to evaluate")
            sleep(60)
        model_dir = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model = GomokuModel(self.config)
        model.load(config_path, weight_path)
        return model, model_dir

    def remove_model(self, model_dir):
        ''' Delete an evaluated candidate model '''
        rc = self.config.resource
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        os.remove(config_path)
        os.remove(weight_path)
        os.rmdir(model_dir)
