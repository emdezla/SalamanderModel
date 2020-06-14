""" Lab 8 """

from multiprocessing import Pool
from run_network import main as run_network
from parse_args import save_plots
import farms_pylog as pylog
from exercise_all import exercise_all
from plot_results import main as plot_results


def main(run_simulations=True, parallel=False):
    """Main function that runs all the exercises."""
    save = save_plots()
    pylog.info('Running network')
    run_network(plot=not save)
    pylog.info('Running simulation exercises')
    arguments = []
    arguments = (
        ['example', '8b', '8c', '8d1', '8d2', '8e', '8f']
        if run_simulations
        else []
    )
    if parallel:
        pool = Pool(processes=4)
        pool.map(exercise_all, [[arg] for arg in arguments])
    else:
        exercise_all(arguments=arguments)
    pylog.info('Plotting simulation results')
    plot_results(plot=not save)


if __name__ == '__main__':
    main()

