import pathlib

import matplotlib.pyplot as plt
import numpy as np

from domains.pancakes import Pancakes, PancakesState
from domains.tile_puzzle import TilePuzzleState, TilePuzzle


def calc_linear_relation(heuristic_values, cost_values):
    heuristic_cost_set = set()
    for i in zip(heuristic_values, cost_values):
        heuristic_cost_set.add(i)
    unique_heuristic_values = [i[0] for i in heuristic_cost_set]
    unique_cost_values = [i[1] for i in heuristic_cost_set]
    a, b = np.polyfit(unique_heuristic_values, unique_cost_values, 1)
    return a, b


def draw_heuristic_as_function_of_cost(heuristic_values, cost_values, xlabel, save_path=None):
    heuristic_values.append(0)
    cost_values.append(0)
    a, b = calc_linear_relation(heuristic_values, cost_values)
    plt.gca().grid(visible=True, which='major', axis='y', zorder=0)
    plt.gca().scatter(heuristic_values, cost_values, s=20, clip_on=False, zorder=3)
    array_for_plot = np.array(sorted(list(set(heuristic_values))), float)
    plt.gca().plot(array_for_plot, np.array(array_for_plot, float) * a + b,
                   linestyle='dashed', color='black', linewidth=5, zorder=4,
                   label=f'y={round(a, 2)}x{"+" if b > 0 else ""}{round(b, 2) if b != 0 else ""}')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.gca().legend(loc="best", handlelength=0, handletextpad=0, fancybox=True, fontsize=15).set_zorder(5)
    plt.xlabel(xlabel)
    plt.ylabel('Optimal Path Cost')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def combine_instances_and_results(data_path, domain_letter):
    state_cost_list = []
    with open(data_path) as f:
        for line in f:
            line = line.strip().split(';')
            if domain_letter == 't':
                instance = TilePuzzleState(tuple(int(i) for i in line[0].split()))
            elif domain_letter == 'p':
                instance = PancakesState(tuple(int(i) for i in line[0].split()))
            else:
                raise Exception('Unknown domain')
            cost = int(line[1])
            state_cost_list.append((instance, cost))
    return state_cost_list


def tile_puzzle_main(ignore_tiles_up_to=0):
    file_path = pathlib.Path.cwd().parent.joinpath('files').joinpath('15_tile_puzzle_state_cost.txt')
    output_path = pathlib.Path.cwd().parent.joinpath('plots').joinpath(f'MD-{ignore_tiles_up_to}')
    state_cost_list = combine_instances_and_results(file_path, 't')
    cost_values = [sc[1] for sc in state_cost_list]
    domain = TilePuzzle(4, 4, ignore_tiles_up_to=ignore_tiles_up_to)
    heuristic_values = [domain.heuristic(sc[0]) for sc in state_cost_list]
    draw_heuristic_as_function_of_cost(heuristic_values, cost_values, f'Heuristic Estimation (MD-{ignore_tiles_up_to})',
                                       save_path=output_path)
    calc_linear_relation(heuristic_values, cost_values)


def pancakes_main(ignore_pancakes_up_to=0):
    file_path = pathlib.Path.cwd().parent.joinpath('files').joinpath('20_pancakes_state_cost.txt')
    output_path = pathlib.Path.cwd().parent.joinpath('plots').joinpath(f'Gap-{ignore_pancakes_up_to}')
    state_cost_list = combine_instances_and_results(file_path, 'p')
    cost_values = [sc[1] for sc in state_cost_list]
    domain = Pancakes(size=20, ignore_pancakes_up_to=ignore_pancakes_up_to)
    heuristic_values = [domain.heuristic(sc[0]) for sc in state_cost_list]
    draw_heuristic_as_function_of_cost(heuristic_values, cost_values,
                                       f'Heuristic Estimation (GAP-{ignore_pancakes_up_to})',
                                       save_path=output_path)


if __name__ == '__main__':
    tile_puzzle_main(0)
    tile_puzzle_main(5)
    tile_puzzle_main(10)
    tile_puzzle_main(13)
    pancakes_main(0)
    pancakes_main(5)
    pancakes_main(10)
    pancakes_main(15)
