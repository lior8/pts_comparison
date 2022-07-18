import math
import pathlib

from tqdm import tqdm

from domains.pancakes import PancakesState, Pancakes
from search.astar_searcher import AstarSearcher
from search.potential_searcher import PotentialSearcher
from search.searcher import Timeout, NoSolution


def setup(instances_id_path, results_path):
    if results_path.is_file() != instances_id_path.is_file():
        raise Exception('Results file cannot exists without instances-ids file and vice versa')
    curr_id = 0
    instances_set = set()
    if instances_id_path.is_file():
        with open(instances_id_path, 'r') as f:
            next(f)
            for line in f:
                instance_id, instance_str, _ = line.strip().split(',')
                instance_id = int(instance_id)
                instance_stack = tuple(int(i) for i in instance_str.split(';'))
                instances_set.add(PancakesState(stack=instance_stack))
                curr_id = instance_id
        if curr_id != 0:
            curr_id += 1
    else:
        with open(instances_id_path, 'w+') as f:
            f.write('instance_id,stack,cost\n')
        with open(results_path, 'w+') as f:
            f.write('instance_id,degradation,bound,h_cost,h_expanded,p_cost,p_expanded\n')
    return curr_id, instances_set


def run_experiment(curr_id, instances_set, instances_id_path, results_path, domain, instances_num=100, timeout=300,
                   quiet=False):
    for _ in tqdm(range(instances_num), total=instances_num, disable=quiet):
        with open(instances_id_path, 'a') as instances_f:
            new_instance = create_instance(domain, instances_set)
            domain.set_heuristic_degradation(0)
            true_cost = AstarSearcher(domain).solve(new_instance, timeout=3600, quiet=True)[0]
            instances_f.write(f'{curr_id},{";".join(str(i) for i in new_instance.stack)},{true_cost}\n')
            with open(results_path, 'a') as results_f:
                for degradation in (0, 0.5, 1, 1.5, 2):
                    domain.set_heuristic_degradation(degradation)
                    for bound_label, bound in ((1, true_cost + 1),
                                               (1.1, math.ceil(true_cost * 1.1)),
                                               (1.25, math.ceil(true_cost * 1.25)),
                                               (1.5, math.ceil(true_cost * 1.5)),
                                               (1.75, math.ceil(true_cost * 1.75)),
                                               (2, math.ceil(true_cost * 2))):
                        h_cost, h_expanded = run_search(domain, new_instance, bound, True, timeout)
                        p_cost, p_expanded = run_search(domain, new_instance, bound, False, timeout)
                        results_f.write(
                            f'{curr_id},{degradation},{bound_label},{h_cost},{h_expanded},{p_cost},{p_expanded}\n')
        curr_id += 1


def run_search(domain, instance, bound, pure_heuristic, timeout):
    pts = PotentialSearcher(domain)
    try:
        cost_found, _ = pts.solve(instance, bound, pure_heuristic, timeout, True)
    except Timeout:
        cost_found = -1
    except NoSolution:
        cost_found = -2
    return cost_found, pts.expanded


def create_instance(domain, instances_set):
    while True:
        new_instance = domain.generate_instances(1, 200, 300)[0]
        if new_instance not in instances_set and not domain.goal_test(new_instance):
            return new_instance


def main():
    num_of_pancakes = 14
    files_dir = pathlib.Path.cwd().parent.joinpath('files')
    instances_id_path = files_dir.joinpath(f'pancakes_instances_ids_{num_of_pancakes}.csv')
    results_path = files_dir.joinpath(f'pancakes_results_{num_of_pancakes}.csv')
    curr_id, instances_set = setup(instances_id_path, results_path)
    domain = Pancakes(size=num_of_pancakes)
    run_experiment(curr_id, instances_set, instances_id_path, results_path, domain, instances_num=100, timeout=300)


if __name__ == '__main__':
    main()
