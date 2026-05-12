"""

envs/Malta_env.py

Malta environment adapted from Sioux_env.py

"""



import logging

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from envs.env import PhaseMap, PhaseSet, RouteMap, RouteSet, TrafficSimulator
from Malta.build_file import gen_rou_file, get_routes, relevant_ss, get_nodes, get_phases, get_flows

import pickle, os

_CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'Malta', 'data', 'malta_cache_v2.pkl')

def _load_or_build_cache():
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    data = {
        'nodes':  get_nodes(),
        'routes': get_routes(),
        'phases': get_phases(),
        'flows':  get_flows(),
    }
    with open(_CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    return data

_cache = _load_or_build_cache()
NODE, NOTLNODE = _cache['nodes']
routes0, rc_num = _cache['routes']
RC_NAMES = ['rc%d' % i for i in range(1, rc_num + 1)]
PHASES, PHASE_NODE_MAP = _cache['phases']






sns.set_color_codes()



STATE_NAMES_SIGNAL = ['wave', 'wait']

STATE_NAMES_ROUTE = ['arrival', 'occupancy']

STATE_ARRIVAL = 1

STATE_OCCUPANCY = 2



'''NODE, NOTLNODE = get_nodes()

routes0, rc_num = get_routes()



PHASES, PHASE_NODE_MAP = get_phases()
'''
RC_NAMES = ['rc%d' % i for i in range(1, rc_num + 1)]


rs_rate = 0.5

thres = 0.05





class MaltaPhase(PhaseMap):

    def __init__(self):

        self.phases = {}

        for key, val in PHASES.items():

            self.phases[key] = PhaseSet(val)





class MaltaRoute(RouteMap):

    def __init__(self):

        self.relevant_rr_map = {}

        self.relevant_rs_map = {}

        self.up_edge_map = {}

        self.down_edge_map = {}

        self.forward_route_map = {}

        self.f_id_map = {}

        self.det_map = {}

        self.routes = {}



        for i, route_set in enumerate(routes0):

            rc_id = 'rc%d' % (i + 1)



            if len(route_set) != 2:

                raise ValueError(

                    "MaltaRoute: expected 2 alternatives per corridor, "

                    "got %d for corridor %d" % (len(route_set), i)

                )



            route_a = route_set[0]

            route_b = route_set[1]



            self.routes[rc_id] = RouteSet(route_set)

            self.up_edge_map[rc_id] = route_a[0] if route_a else ''



            down = {}

            for j, eid in enumerate(route_a):

                down[eid] = rs_rate ** j

            for j, eid in enumerate(route_b):

                if eid not in down:

                    down[eid] = rs_rate ** j

                else:

                    down[eid] = max(down[eid], rs_rate ** j)

            self.down_edge_map[rc_id] = down



            self.forward_route_map[rc_id] = [route_a, route_b]



            self.det_map[rc_id] = [

                route_a[-1] + '_0',

                route_b[-1] + '_0'

            ]



            self.f_id_map[rc_id] = str(i + 1)



            self.relevant_rs_map[rc_id] = {}

            all_edges = route_a + route_b

            for j, eid in enumerate(all_edges):

                weight = rs_rate ** j

                if weight < thres:

                    break

                if eid not in self.relevant_rs_map[rc_id]:

                    self.relevant_rs_map[rc_id][eid] = weight

                else:

                    self.relevant_rs_map[rc_id][eid] = max(

                        self.relevant_rs_map[rc_id][eid], weight

                    )



            self.relevant_rr_map[rc_id] = []



        self.relevant_sr_map = {n: {} for n in NODE}





class MaltaController:

    def __init__(self, node_names, rc_names):

        self.name = 'greedy'

        self.node_names = node_names

        self.rc_names = rc_names



    def forward(self, obs):

        actions = []

        for ob, node_name in zip(obs, self.node_names):

            actions.append(self.greedy(ob, node_name))

        return actions



    def greedy(self, ob, node_name):

        if len(ob) < 6:

            return 0

        flows = [ob[0] + ob[3], ob[2] + ob[5], ob[1] + ob[4],

                 ob[1] + ob[2], ob[4] + ob[5]]

        return np.argmax(np.array(flows))





class MaltaEnv(TrafficSimulator):
    def init_data(self, is_record, output_path, diff=None):
        self.is_record = is_record
        self.output_path = output_path
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
            self.trip_data = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def terminate(self):
        self.sim.close()

    def reset(self, gui=False, test_ind=0, load=False, separate=False):
        self._reset_state()
        if self.train_mode:
            seed = self.seed
        else:
            seed = self.test_seeds[test_ind]
        self._init_sim(seed, gui=gui)
        self.cur_sec = 0
        self.cur_episode += 1
        if self.agent == 'ma2c':
            self.update_fingerprint(self._init_policy())
        self._init_sim_traffic()
        self.seed += 1
        return self._get_state(reset=True)

    def output_data(self):
        if not self.is_record:
            return
        import pandas as pd
        pd.DataFrame(self.control_data).to_csv(self.output_path + (self.name + '_' + self.problem + '_control.csv'))
        pd.DataFrame(self.traffic_data).to_csv(self.output_path + (self.name + '_' + self.problem + '_traffic.csv'))
        pd.DataFrame(self.trip_data).to_csv(self.output_path + (self.name + '_' + self.problem + '_trip.csv'))

    def init_data(self, is_record, output_path, diff=None):
        self.is_record = is_record
        self.output_path = output_path
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
            self.trip_data = []


    def __init__(self, config, problem, output_path='', is_record=False, port=0):
        self.demand_flows = _cache['flows']
        super().__init__(config, problem, output_path, is_record, port=port)


    def _get_node_names(self):

        """Return only TL nodes that exist in phase_node_map."""

        all_nodes = self.sim.trafficlight.getIDList()

        valid = [n for n in all_nodes if n in self.phase_node_map]

        skipped = [n for n in all_nodes if n not in self.phase_node_map]

        for n in skipped:

            logging.info('Skipping TL node not in phase_node_map: %s' % n)

        return valid



    def _get_node_phase_id(self, node_name):

        """Return phase_id for node, finding best lane-count match if needed."""

        if node_name in self.phase_node_map:

            return self.phase_node_map[node_name]

        fallback = next(iter(self.phase_node_map))

        logging.warning('node %s not in phase_node_map, using fallback %s'

                        % (node_name, fallback))

        return fallback



    def _find_matching_phase(self, lane_num):

        """Find a phase_id whose encoded lane count matches lane_num."""

        # exact match first

        for phase_id in self.phase_map.phases:

            if self.phase_map.get_lane_num(phase_id) == lane_num:

                return phase_id

        # closest match

        best_id = None

        best_diff = float('inf')

        for phase_id in self.phase_map.phases:

            diff = abs(self.phase_map.get_lane_num(phase_id) - lane_num)

            if diff < best_diff:

                best_diff = diff

                best_id = phase_id

        return best_id



    def _get_node_state_num(self, node):

        lane_num = len(node.lanes_in)

        current_phase_id = node.phase_id



        if self.phase_map.get_lane_num(current_phase_id) != lane_num:

            best_phase = self._find_matching_phase(lane_num)

            if best_phase:

                logging.info(

                    'node %s: lane count %d does not match phase %s (%d lanes), switching to phase %s'

                    % (node.name, lane_num, current_phase_id,

                    self.phase_map.get_lane_num(current_phase_id), best_phase)

                )

                node.phase_id = best_phase



        return len(node.ilds_in)



    def _get_rc_route_num(self, rc_name):

        return 2



    def _get_rc_names(self):

        return RC_NAMES



    def _get_rc_state_num(self):
        return STATE_ARRIVAL

    # ADD THIS RIGHT HERE ↓
    def _init_action_space(self):
        n_a_ls_signal = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase_id = self._get_node_phase_id(node_name)
            node.phase_id = phase_id
            node.n_a = self.phase_map.get_phase_num(phase_id)
            n_a_ls_signal.append(node.n_a)

        n_a_ls_route = []
        for rc_name in self.rc_names:
            rc = self.rcs[rc_name]
            rc.route_num = self._get_rc_route_num(rc_name)
            rc.n_a = rc.route_num
            n_a_ls_route.append(rc.n_a)

        if self.problem == 'signal_route':
            self.n_a_ls = n_a_ls_signal + n_a_ls_route
        elif self.problem == 'signal':
            self.n_a_ls = n_a_ls_signal
        elif self.problem == 'route':
            self.n_a_ls = n_a_ls_route

        # Use max instead of np.prod to avoid combinatorial explosion (e.g. 3^190)
        self.n_a = max(self.n_a_ls)

    



    def _init_relevant_ss_map(self):

        return relevant_ss()



    def _init_map(self):

        self.phase_node_map = PHASE_NODE_MAP

        self.phase_map = MaltaPhase()

        self.route_map = MaltaRoute()

        self.relevant_ss_map = self._init_relevant_ss_map()

        self.relevant_rs_map = self.route_map.relevant_rs_map

        self.relevant_sr_map = self.route_map.relevant_sr_map

        self.relevant_rr_map = self.route_map.relevant_rr_map

        self.up_edge_map = self.route_map.up_edge_map

        self.down_edge_map = self.route_map.down_edge_map

        self.forward_route_map = self.route_map.forward_route_map

        self.f_id_map = self.route_map.f_id_map

        self.det_map = self.route_map.det_map

        self.state_names_signal = STATE_NAMES_SIGNAL

        self.state_names_route = STATE_NAMES_ROUTE

        # clean relevant_rs_map to only keep TL node keys

        tl_set = set(NODE)

        for rc in self.relevant_rs_map:

            self.relevant_rs_map[rc] = {

                k: v for k, v in self.relevant_rs_map[rc].items()

                if k in tl_set

            }



    def _init_sim_config(self, seed, load):

        return gen_rou_file(

            self.data_path,

            self.demand_flows,

            seed=seed,

            thread=self.sim_thread,

            load=load

        )



    def plot_stat(self, rewards):

        self.state_stat['reward'] = rewards

        for name, data in self.state_stat.items():

            fig = plt.figure(figsize=(8, 6))

            plot_cdf(data)

            plt.ylabel(name)

            fig.savefig(self.output_path + self.name + '_' + name + '.png')





def plot_cdf(X, c='b', label=None):

    sorted_data = np.sort(X)

    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

    plt.plot(sorted_data, yvals, color=c, label=label)