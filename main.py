import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Force CPU-only by default. Originally needed because the per-call
# Python<->GPU copy cost per agent dominated for batch-size-1 LSTM
# inference. After the IA2C.forward/backward batching refactor (one
# sess.run for all 190 agents), the GPU may actually win — try toggling
# this off (comment the next line) once a CPU-only run is verified.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Enable Intel oneDNN CPU math kernels (typically 2-4x speedup on CPU
# matmul/conv/LSTM compared to the stock TF kernels).
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Let oneDNN/OpenMP use every core. os.cpu_count() returns logical cores.
_n_cores = str(os.cpu_count() or 8)
os.environ.setdefault('OMP_NUM_THREADS', _n_cores)
os.environ.setdefault('KMP_BLOCKTIME', '0')  # don't spin between calls
os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import configparser
import tensorflow as tf
# Report what TF can see — useful to confirm oneDNN + GPU detection.
print('>>> TF version:', tf.__version__, '| logical CPUs:', _n_cores, flush=True)
try:
    _gpus = tf.config.list_physical_devices('GPU')
    print('>>> TF GPUs visible:', _gpus or 'none (CPU-only run)', flush=True)
except Exception as _e:
    print('>>> TF GPU probe failed:', _e, flush=True)
tf = tf.compat.v1
tf.disable_v2_behavior()
import threading

from envs.Malta_env import MaltaEnv as SiouxEnv, MaltaController as SiouxController
from agents.models import MA2C
from utils import *  

def init_env(config, port=0, naive_policy=False):
    if not naive_policy:
        return SiouxEnv(config, problem, port=port)
    else:
        env = SiouxEnv(config, problem, port=port)
        policy = SiouxController(env.node_names, env.rc_names)
        return env, policy


def get_parameter(scenario):
    config_dir = default_config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)
    env = SiouxEnv(config['ENV_CONFIG'],problem, port=0)
    env.get_parameter(scenario)
    env.terminate()

def train():
    base_dir = default_base_dir
    dirs = init_dir(base_dir)
    store_dir=default_store_dir
    dirs2=init_dir(store_dir)
    init_log(dirs2['log'])
    config_dir = default_config_dir
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)
    test_mode='no_test'
    in_test, post_test = init_test_flag(test_mode)

    # init env
    print(">>> Initializing env (launches SUMO for dimensions)...", flush=True)
    env = init_env(config['ENV_CONFIG'], port=0)
    logging.info('Training: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))

    print(">>> Building MA2C model...", flush=True)

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    
    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')

    if problem == 'signal_route':
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, total_step,
                     config['MODEL_CONFIG'], problem, agent_type=env.agent_type, n_o_ls=env.n_o_ls, seed=seed)
    else:
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, total_step,
                     config['MODEL_CONFIG'], problem, seed=seed)


    summary_writer =tf.summary.FileWriter(dirs2['log'])
    trainer = Trainer(env, model,problem, global_counter, summary_writer, in_test, output_path=dirs['data'])
    print(">>> Starting trainer.run() — first episode loading SUMO...", flush=True)
    trainer.run()

    # post-training test
    if post_test:
        tester = Tester(env, model, problem,global_counter, summary_writer, dirs['data'])
        tester.run_offline(dirs['data'])

    # save model
    final_step = global_counter.cur_step
    logging.info('Training: save final model at step %d ...' % final_step)
    model.save(dirs2['model'], final_step)


def evaluate_fn(agent, output_dir, seeds, port, demo, policy_type):
    # load config file for env
    config_dir = default_config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env, greedy_policy = init_env(config['ENV_CONFIG'], port=port, naive_policy=True)
    logging.info('Evaluation: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))
    env.init_test_seeds(seeds)
    if separate_train==True:
        n_s_ls=[46, 36, 14, 22, 38, 38, 30, 22, 28, 34, 28, 30, 12, 14, 14, 28, 24]
        n_a_ls=[5, 4, 2, 2, 4, 4, 2, 2, 2, 4, 2, 4, 2, 2, 2, 4, 2]
        n_w_ls=[10, 8, 6, 6, 8, 8, 6, 6, 6, 8, 6, 8, 6, 6, 6, 8, 6]
        n_f_ls=[11, 7, 3, 6, 9, 9, 8, 6, 7, 6, 7, 7, 1, 3, 3, 5, 7]
        model_signal = MA2C(n_s_ls, n_a_ls, n_w_ls, n_f_ls, 0,
                             config['MODEL_CONFIG'], 'signal', agent_type=env.agent_type, n_o_ls=env.n_o_ls)
        n_s_ls=[1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1]
        n_a_ls=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        n_w_ls=[6, 7, 10, 7, 6, 10, 7, 9, 7, 10, 6, 9]
        n_f_ls=[0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]
        model_route=MA2C(n_s_ls, n_a_ls, n_w_ls, n_f_ls, 0,
                             config['MODEL_CONFIG'], 'route', agent_type=env.agent_type, n_o_ls=env.n_o_ls)
        if not model_signal.load(default_base_dir+ '/signal/model/'):
            return
        if not model_route.load(default_base_dir+'/route/model/'):
            return
        env.agent=agent
        evaluator = Evaluator_separate(env, model_signal,model_route, output_dir, demo=demo, policy_type=policy_type)
        evaluator.run()
        return

    if problem == 'signal_route':
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, 0,
                             config['MODEL_CONFIG'], problem, agent_type=env.agent_type, n_o_ls=env.n_o_ls)
    else:
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, 0,
                             config['MODEL_CONFIG'], problem)

    if not model.load(default_store_dir + '/model/'):
        return

    env.agent = agent
    evaluator = Evaluator(env, model, output_dir, demo=demo, policy_type=policy_type)
    if diff==False:
        evaluator.run()
    else:
        if compliance==True:
            evaluator.run_diff_com()
        elif load==True:
            evaluator.run_diff_load()


def run_greedy():
    """Run one full episode with the MaltaController (greedy/naive policy) and save metrics."""
    import pandas as pd

    config = configparser.ConfigParser()
    config.read(default_config_dir)
    env, policy = init_env(config['ENV_CONFIG'], port=0, naive_policy=True)

    store_dir = default_store_dir
    dirs = init_dir(store_dir, pathes=['greedy_data', 'log'])
    init_log(dirs['log'])

    seeds_str = config.get('ENV_CONFIG', 'test_seeds', fallback='10000,20000')
    seeds = [int(s) for s in seeds_str.split(',')]
    env.init_test_seeds(seeds)
    env.train_mode = False

    data = []
    for test_ind in range(env.test_num):
        ob = env.reset(gui=False, test_ind=test_ind)
        done = False
        total_reward = 0
        total_arrived = 0
        total_departed = 0
        total_wait = 0
        total_travel = 0
        signal_n_a = env.n_a_ls[:len(env.node_names)]
        step_count = 0
        while not done:
            actions = policy.forward(ob)
            # clip each signal action to its node's valid phase range
            actions = [min(a, n - 1) for a, n in zip(actions, signal_n_a)]
            # route agents get a fixed action (stay on current route) when using greedy signal only
            full_actions = actions + [0] * len(env.rc_names)
            ob, reward, done, global_reward, arrived, departed, wait, travel, _ = env.step(full_actions)
            step_count += 1
            if step_count % 72 == 0:
                print('  episode %d | sim time %ds / %ds | arrived %d' % (
                    test_ind, step_count * 5, 3600, total_arrived), flush=True)
            total_reward += global_reward
            total_arrived += arrived
            total_departed += departed
            total_wait += wait
            total_travel += travel

        env.terminate()
        if total_departed > 0:
            avg_wait = total_wait / total_departed
            avg_travel = total_travel / total_departed
        else:
            avg_wait = avg_travel = float('nan')

        log = {
            'agent': 'greedy',
            'test_id': test_ind,
            'reward': total_reward,
            'total arrived': total_arrived,
            'total departed': total_departed,
            'average waiting time': avg_wait,
            'average travel time': avg_travel,
        }
        data.append(log)
        logging.info('Greedy test %d: avg_wait=%.2f s, avg_travel=%.2f s' % (test_ind, avg_wait, avg_travel))

    out_path = os.path.join(dirs['greedy_data'], 'greedy_baseline.csv')
    pd.DataFrame(data).to_csv(out_path, index=False)
    logging.info('Greedy baseline saved to %s' % out_path)


def evaluate():
    store_dir = default_store_dir
    dirs = init_dir(store_dir, pathes=['eva_data', 'eva_log'])
    init_log(dirs['eva_log'])
    agents={'ma2c'}
    seeds = ','.join([str(i) for i in range(10000, 100001, 10000)])
    policy_type = 'default'
    logging.info('Evaluation: policy type: %s, random seeds: %s' % (policy_type, seeds))
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    threads = []
    for i, agent in enumerate(agents):
        demo=False
        thread = threading.Thread(target=evaluate_fn,
                                  args=(agent, dirs['eva_data'], seeds, i, demo, policy_type))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


default_config_dir = './config/config_ma2c_Malta.ini'
scenario = 'Malta'

option='evaluate' #'evaluate','get_parameter'*3

separate_train=False

diff=False
compliance=False
load=False

problem='signal_route'
default_base_dir = scenario
if separate_train==True:
    default_store_dir = default_base_dir + '/signal_route(separate_training)'
else:
    default_store_dir=default_base_dir+ '/' + problem


if __name__ == '__main__':
    if option == 'train':
        train()
    elif option == 'evaluate':
        evaluate()
    elif option == 'greedy':
        run_greedy()
    elif option == 'get_parameter':
        get_parameter(scenario)
