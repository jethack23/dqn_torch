mahn = 10000


class MemoryConfig:
    mem_dir = "save/memory/"
    memory_size = 50 * mahn


class ModelConfig:
    model_name = "DQN"
    in_height = 1
    in_width = 4

    batch_size = 32
    history_length = 1

    cnn_archi = []
    fc_archi = [32, 64]
    output_size = 2

    log_dir = "save/tensorboard/"
    ckpt_dir = "save/ckpt/"


class EnvironmentConfig:
    # env_name = "BreakoutDeterministic-v4"
    env_name = "Breakout-v0"

    max_reward = 1.
    min_reward = -1.


class TrainConfig:
    max_step = 500 * mahn

    initial_exploration = 1.
    final_exploration = 0.1
    final_exploration_step = 50 * mahn

    test_exploration = 0.05
    test_play_num = 100

    replay_start_size = 5 * mahn
    fixed_net_update_frequency = mahn

    replay_frequency = 4
    action_repeat = 1

    no_op_max = 30

    df = 0.99

    lr = 0.00025
    lr_min = 0.00025
    lr_decay = 0.96
    lr_decay_step = 5 * mahn

    save_step = 10 * mahn
    summarize_step = 1 * mahn
    load_model = True


class Config(MemoryConfig, ModelConfig, EnvironmentConfig, TrainConfig):
    load_ckpt = 1
    test = 0
    train = 1
    render = 0
