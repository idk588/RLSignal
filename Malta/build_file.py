# -*- coding: utf-8 -*-
"""
Malta/build_file.py
Replaces Sioux/data/build_file.py for the Malta road network.
Reads directly from malta.net.xml instead of CSV files.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict, deque

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NET_FILE  = os.path.join(DATA_DIR, 'malta.net.xml')

MAX_CAR_NUM  = 30
SPEED_LIMIT  = 50
time_length  = 3600


# ── helpers ───────────────────────────────────────────────────────────────────

# Process-wide cache of the parsed net + derived structures.
# Parsing malta.net.xml is expensive (~2 MB, 400k+ edges), so we keep one copy.
_NET_CACHE = {}


def _parse_net():
    if 'root' not in _NET_CACHE:
        _NET_CACHE['root'] = ET.parse(NET_FILE).getroot()
    return _NET_CACHE['root']


def _lane_allows_passenger(lane_elem, edge_disallow, edge_allow):
    allow = lane_elem.attrib.get('allow', edge_allow)
    disallow = lane_elem.attrib.get('disallow', edge_disallow)
    if allow:
        return 'passenger' in allow.split()
    if disallow:
        return 'passenger' not in disallow.split()
    return True  # SUMO default: passenger allowed


def _build_net_index():
    """One-shot scan of malta.net.xml: identify passenger-allowed edges, build
    an edge connectivity graph from <connection> elements, and collect
    traffic-light junctions. Cached for the life of the process."""
    if _NET_CACHE.get('index_built'):
        return
    root = _parse_net()

    passenger_edges = set()
    edge_from = {}
    edge_to = {}
    for edge in root.findall('edge'):
        if edge.attrib.get('function') == 'internal':
            continue
        eid = edge.attrib.get('id', '')
        if not eid or eid.startswith(':'):
            continue
        e_disallow = edge.attrib.get('disallow', '')
        e_allow = edge.attrib.get('allow', '')
        lanes = edge.findall('lane')
        if not lanes:
            continue
        if any(_lane_allows_passenger(l, e_disallow, e_allow) for l in lanes):
            passenger_edges.add(eid)
            edge_from[eid] = edge.attrib.get('from', '')
            edge_to[eid] = edge.attrib.get('to', '')

    edge_graph = defaultdict(set)
    for conn in root.findall('connection'):
        fr = conn.attrib.get('from', '')
        to = conn.attrib.get('to', '')
        if fr in passenger_edges and to in passenger_edges:
            edge_graph[fr].add(to)

    tl_junctions = set()
    for j in root.findall('junction'):
        if j.attrib.get('type') == 'traffic_light':
            tl_junctions.add(j.attrib.get('id', ''))

    _NET_CACHE['passenger_edges'] = passenger_edges
    _NET_CACHE['edge_from'] = edge_from
    _NET_CACHE['edge_to'] = edge_to
    _NET_CACHE['edge_graph'] = edge_graph
    _NET_CACHE['tl_junctions'] = tl_junctions
    _NET_CACHE['index_built'] = True


def _route_is_valid(edge_seq):
    """A route is valid if every edge allows passenger and consecutive edges
    have a SUMO connection between them."""
    if not edge_seq:
        return False
    _build_net_index()
    passenger_edges = _NET_CACHE['passenger_edges']
    edge_graph = _NET_CACHE['edge_graph']
    for e in edge_seq:
        if e not in passenger_edges:
            return False
    for i in range(len(edge_seq) - 1):
        if edge_seq[i + 1] not in edge_graph.get(edge_seq[i], set()):
            return False
    return True


def _bfs_path(src, max_depth=15, want_tl_count=2):
    """BFS from src; return the first path with >= want_tl_count TL crossings
    and length >= 4. Returns None if no such path exists."""
    _build_net_index()
    edge_graph = _NET_CACHE['edge_graph']
    edge_to = _NET_CACHE['edge_to']
    tl_junctions = _NET_CACHE['tl_junctions']
    visited = {src: None}
    queue = deque([(src, 0, 0)])
    while queue:
        cur, depth, tl_count = queue.popleft()
        if tl_count >= want_tl_count and depth >= 4:
            path = []
            x = cur
            while x is not None:
                path.append(x)
                x = visited[x]
            return list(reversed(path))
        if depth >= max_depth:
            continue
        for nxt in edge_graph.get(cur, ()):
            if nxt in visited:
                continue
            visited[nxt] = cur
            new_tl = tl_count + (1 if edge_to.get(nxt) in tl_junctions else 0)
            queue.append((nxt, depth + 1, new_tl))
    return None


def _validate_or_fix_routes(routes):
    """For every (corridor, alternative) edge sequence, keep it if valid;
    otherwise replace it with a BFS-discovered passenger-allowed path that
    crosses at least 2 TL junctions. Each corridor is guaranteed to end
    with exactly 2 alternatives (env requires this)."""
    _build_net_index()
    passenger_edges = _NET_CACHE['passenger_edges']
    sources_pool = sorted(passenger_edges)
    rng = np.random.default_rng(123)

    fixed = []
    for i, route_set in enumerate(routes):
        new_set = []
        for r in route_set:
            if _route_is_valid(r):
                new_set.append(r)
                continue
            # try to extend the original first edge if it's at least passenger-allowed
            seed_src = r[0] if (r and r[0] in passenger_edges) else None
            new_path = _bfs_path(seed_src) if seed_src else None
            if new_path is None:
                # last resort: pick random passenger-allowed sources
                for _ in range(50):
                    cand = sources_pool[rng.integers(0, len(sources_pool))]
                    new_path = _bfs_path(cand)
                    if new_path:
                        break
            if new_path is None:
                # absolute last resort: a single passenger edge — SUMO will
                # accept it (single-edge route) even if it doesn't cross a TL
                new_path = [sources_pool[rng.integers(0, len(sources_pool))]]
            print('[Malta route fix] corridor %d alt %d: replaced %r with %d-edge path'
                  % (i + 1, len(new_set) + 1, r, len(new_path)))
            new_set.append(new_path)
        # env requires exactly 2 alternatives per corridor
        while len(new_set) < 2:
            new_set.append(new_set[0])
        fixed.append(new_set[:2])
    return fixed


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


# ── required by Malta_env.py ──────────────────────────────────────────────────

def get_nodes(build=False):
    root = _parse_net()

    # Get valid TL IDs — must have >= 2 phases
    valid_tl_ids = set()
    for tl in root.findall('tlLogic'):
        tl_id  = tl.attrib.get('id')
        states = [p.attrib['state'] for p in tl.findall('phase') if 'state' in p.attrib]
        if tl_id and len(states) >= 2:
            valid_tl_ids.add(tl_id)

    tl_nodes, non_tl_nodes = [], []
    for j in root.findall('junction'):
        jid   = j.attrib.get('id', '')
        jtype = j.attrib.get('type', '')
        if jid.startswith(':'):
            continue
        if jtype == 'traffic_light':
            if jid in valid_tl_ids:
                tl_nodes.append(jid)
            else:
                # treat excluded TL junctions as non-TL
                non_tl_nodes.append(jid)
        elif jtype in ('priority', 'right_before_left', 'unregulated'):
            non_tl_nodes.append(jid)

    return tl_nodes, non_tl_nodes



def get_phases():
    root = _parse_net()
    phases = {}
    phase_node_map = {}
    for tl in root.findall('tlLogic'):
        tl_id  = tl.attrib.get('id')
        states = [p.attrib['state'] for p in tl.findall('phase') if 'state' in p.attrib]
        # SKIP junctions with only 1 phase — model needs at least 2 actions
        if tl_id and len(states) >= 2:
            phases[tl_id]          = states
            phase_node_map[tl_id]  = tl_id
    return phases, phase_node_map


def get_routes():
    """
    Route-choice alternatives for Malta corridors using real SUMO edge IDs.

    Corridor 1 — Route choice at junction 6484114392 (two alternatives):
      Route A: via 1657799943 and 301487637 (longer alternative)
      Route B: direct to 6484114393

    Corridors 2-4 — Bidirectional arterial pairs
    """
    routes = [
        # Corridor 1: main route choice point
        # Route A: 54444410#2 -> 1264291443 -> 33698427#1
        # Route B: 824060363#1 (direct)
        [
            ['54444410#2', '1264291443', '33698427#1'],
            ['824060363#1'],
        ],

        # Corridor 2: bidirectional arterial
        [
            ['-1456229447#1'],
            ['1456229447#1'],
        ],

        # Corridor 3: bidirectional arterial
        [
            ['-27446664#1'],
            ['27446664#1'],
        ],

        # Corridor 4: bidirectional arterial
        [
            ['-743322159'],
            ['743322159'],
        ],
    ]

    # Auto-validate and repair: any route whose edges aren't connected in the
    # network (or aren't passenger-allowed) gets replaced with a BFS-discovered
    # path. This stops SUMO from erroring out on "no valid route" and from
    # silently dropping flows on disallowed edges.
    routes = _validate_or_fix_routes(routes)
    rc_num = sum(len(r) - 1 for r in routes)
    return routes, rc_num


def relevant_ss(build=False):
    """
    Return a dict mapping each TL node to its directly connected TL neighbours.
    Built automatically from the edges in malta.net.xml.
    """
    root = _parse_net()
    tl_nodes, _ = get_nodes()
    tl_set = set(tl_nodes)

    relevant_ss_map = {n: [] for n in tl_nodes}
    for edge in root.findall('edge'):
        if edge.attrib.get('function') == 'internal':
            continue
        frm = edge.attrib.get('from', '')
        to  = edge.attrib.get('to', '')
        if frm in tl_set and to in tl_set:
            if to not in relevant_ss_map[frm]:
                relevant_ss_map[frm].append(to)
    return relevant_ss_map


def get_flows(build=False):
    """
    Vehicles/hour per corridor calibrated to Malta peak-hour conditions.
    Index matches corridor order in get_routes().
    Corridor 1: main route-choice point (high demand, Msida-Valletta type)
    Corridors 2-4: bidirectional arterial flows
    """
    return [600, 300, 400, 350]


# ── route/config file generation ─────────────────────────────────────────────

def _get_edge_ids_for_route(edge_sequence):
    """Routes are already edge IDs — return them directly."""
    return edge_sequence


def output_flows(demand_flows, seed=None, load=False):
    routes, _ = get_routes()
    demand_begin = 0
    demand_end   = 3600
    demand_gap   = 360

    ratios = [
        [0.3, 0.5, 0.4, 0.3, 0.8, 0.9, 1.0, 0.9, 0.7, 0.2],
        [0.3, 0.2, 0.4, 0.3, 0.5, 0.5, 0.8, 0.8, 0.9, 1.0],
        [0.2, 0.3, 0.5, 0.4, 0.7, 0.8, 1.0, 0.9, 0.6, 0.3],
        [0.4, 0.5, 0.6, 0.5, 0.8, 0.9, 1.0, 0.8, 0.5, 0.2],
    ]

    if seed is not None:
        np.random.seed(seed)

    str_flows = '<routes>\n'
    str_flows += '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>\n'

    # Define route alternatives — edges already provided directly
    for i, route_set in enumerate(routes):
        for k, edge_seq in enumerate(route_set):
            edges = ' '.join(edge_seq)
            str_flows += '    <route id="route_%d_%d" edges="%s"/>\n' % (i+1, k+1, edges)

    # Generate flows per time interval
    for j in range(10):
        t_begin = j * demand_gap
        t_end   = (j + 1) * demand_gap
        for i, route_set in enumerate(routes):
            flow_val = ratios[i % len(ratios)][j] * demand_flows[i % len(demand_flows)]
            if load:
                flow_val = int(flow_val * load)
            str_flows += (
                '    <flow id="flow_%d_%d" route="route_%d_1" '
                'begin="%d" end="%d" vehsPerHour="%.1f" type="car"/>\n'
                % (i+1, j+1, i+1, t_begin, t_end, flow_val)
            )

    str_flows += '</routes>\n'
    return str_flows


def output_config(thread=None, add_file=None):
    if thread is None:
        rou_file = 'exp.rou.xml'
    else:
        rou_file = 'exp_%d.rou.xml' % int(thread)

    add_line = ''
    if add_file:
        add_line = '        <additional-files value="%s"/>\n' % add_file

    cfg = (
        '<configuration>\n'
        '    <input>\n'
        '        <net-file value="malta.net.xml"/>\n'
        '        <route-files value="%s"/>\n'
        '%s'
        '    </input>\n'
        '    <time>\n'
        '        <begin value="0"/>\n'
        '        <end value="%d"/>\n'
        '    </time>\n'
        '    <processing>\n'
        '        <time-to-teleport value="-1"/>\n'
        '    </processing>\n'
        '</configuration>\n'
    ) % (rou_file, add_line, time_length)
    return cfg


def output_additional(data_path):
    """
    Generate E2 lane-area detectors only on lanes that lead INTO traffic-light
    junctions. The env's _init_agents only ever queries detectors that match
    trafficlight.getControlledLanes(), so emitting one per lane in the entire
    network (~120k detectors) was massive wasted work at SUMO startup.

    Also: dropped freq=1 + per-second file output (was ~430M lines/episode).
    The env reads detector state via TraCI getLastStep* calls, which work
    independently of the aggregation period.
    """
    root = _parse_net()

    tl_junctions = set()
    for j in root.findall('junction'):
        if j.attrib.get('type') == 'traffic_light':
            tl_junctions.add(j.attrib.get('id', ''))

    str_add = '<additional>\n'
    count = 0
    for edge in root.findall('edge'):
        if edge.attrib.get('function') == 'internal':
            continue
        eid = edge.attrib.get('id', '')
        if not eid or eid.startswith(':'):
            continue
        # Only lanes feeding a TL junction matter for the agents
        if edge.attrib.get('to', '') not in tl_junctions:
            continue
        for lane in edge.findall('lane'):
            lane_id = lane.attrib.get('id', '')
            lane_len = float(lane.attrib.get('length', 100))
            det_len = min(lane_len - 0.1, 100)
            if det_len <= 0:
                det_len = 1.0
            # SUMO requires both `file` and a period; we set period to the
            # full episode length so only ~one row per detector is ever
            # written (442 rows total, negligible). Per-step queries via
            # TraCI getLastStep* still work regardless of period.
            str_add += (
                '    <laneAreaDetector id="%s_0" lane="%s" '
                'pos="0" length="%.2f" period="3600" file="ild_out.xml"/>\n'
                % (lane_id, lane_id, det_len)
            )
            count += 1
    str_add += '</additional>\n'
    print('[Malta detectors] emitted %d laneAreaDetectors (TL-incoming lanes only)' % count)
    return str_add


def gen_rou_file(path, demand_flows, seed=None, thread=None, load=False):
    if thread is None:
        flow_file = 'exp.rou.xml'
    else:
        flow_file = 'exp_%d.rou.xml' % int(thread)

    write_file(path + flow_file, output_flows(demand_flows, seed=seed, load=load))

    # generate detector additional file
    add_file = path + 'exp.add.xml'
    write_file(add_file, output_additional(path))

    sumocfg_file = path + ('exp_%d.sumocfg' % thread)
    write_file(sumocfg_file, output_config(thread=thread, add_file='exp.add.xml'))
    return sumocfg_file