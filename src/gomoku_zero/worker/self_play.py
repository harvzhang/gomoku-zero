import os
from datetime import datetime
from logging import getLogger
from time import time

from gomoku_zero.agent.player_gomoku import GomokuPlayer, MCTSMemory
from gomoku_zero.config import Config
from gomoku_zero.env.gomoku_env import GomokuEnv, Winner, Player
from gomoku_zero.lib import tf_util
from gomoku_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from gomoku_zero.lib.model_helpler import load_best_model_weight, save_as_best_model, \
    reload_best_model_weight_if_changed

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.2)
    return SelfPlayWorker(config, env=GomokuEnv()).start()


class SelfPlayWorker:
    def __init__(self, config: Config, env=None, model=None):
        """
        Generates play data by performing self-play
        :param config:
        :param Connect4Env|None env:
        :param gomoku_zero.agent.model_connect4.Connect4Model|None model:
        """
        self.config = config
        self.model = model
        self.env = env     # type: Connect4Env
        self.black = None  # type: Connect4Player
        self.white = None  # type: Connect4Player
        self.buffer = []

    def start(self):
        ''' Entry point to the self play thread '''

        if self.model is None:
            self.model = self.load_model()

        self.buffer = []
        idx = 1

        while True:
            start_time = time()
            # play a game
            env = self.start_game(idx)
            end_time = time()
            logger.debug(f"game {idx} time={end_time - start_time} sec, "
                         f"turn={env.turn}:{env.observation} - Winner:{env.winner}")

            # update the model if new best model is available
            if (idx % self.config.play_data.nb_game_in_file) == 0:
                reload_best_model_weight_if_changed(self.model)
            idx += 1

    def start_game(self, idx):
        '''
        Plays a single game with the best model
        :param idx: index of the game
        :return:
        '''

        self.env.reset()
        memory = None
        if self.config.trainer.mc_rave:
            memory = MCTSMemory(self.config)
        self.black = GomokuPlayer(self.config, self.model, mem=memory)
        self.white = GomokuPlayer(self.config, self.model, mem=memory)

        # play a game until termination
        while not self.env.done:
            if self.env.player_turn() == Player.black:
                action = self.black.action(self.env.board, self.env.turn)
            else:
                action = self.white.action(self.env.board, self.env.turn)
            self.env.step(action)
        self.finish_game()
        self.save_play_data(write=idx % self.config.play_data.nb_game_in_file == 0)
        # remove old play data
        self.remove_play_data()
        return self.env

    def save_play_data(self, write=True):
        '''
        Saves the state, action, rewards data into file
        :return:
        '''

        data = self.black.moves + self.white.moves
        self.buffer += data

        if not write:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        write_game_data_to_file(path, self.buffer)
        self.buffer = []

    def remove_play_data(self):
        ''' Delete old play file if exceed max_file_num '''
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        for i in range(len(files) - self.config.play_data.max_file_num):
            os.remove(files[i])

    def finish_game(self):
        ''' Assign the game reward at end game '''
        if self.env.winner == Winner.black:
            black_win = 1
        elif self.env.winner == Winner.white:
            black_win = -1
        else:
            black_win = 0

        self.black.finish_game(black_win)
        self.white.finish_game(-black_win)

    def load_model(self):
        ''' Load the current best model '''
        from gomoku_zero.agent.model_gomoku import GomokuModel
        model = GomokuModel(self.config)
        if not load_best_model_weight(model):
            model.build(self.config.model.init_res_layer_num)
            save_as_best_model(model)
        return model



