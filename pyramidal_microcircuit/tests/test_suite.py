import sys
import nest
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import traceback
from tests01_neuron_dynamics import *
from tests02_plasticity import *
sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")
import utils  # nopep8
from params import *  # nopep8

if __name__ == "__main__":

    if len(sys.argv) > 1:
        classes = [eval(sys.argv[1])]
    else:
        classes = [FilteredInputCurrent, CurrentConnection, SingleCompartmentDynamics, TargetCurrent, NetworkDynamics, PlasticityBasal, PlasticityApical, NetworkPlasticity]


    root, imgdir, datadir = utils.setup_simulation(
        "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/tests/runs")

    sim_params["record_interval"] = 0.1
    sim_params["recording_backend"] = "memory"

    plot_all_runs = True

    for use_spiking_neurons in [False, True]:
        spiking_str = 'spiking' if use_spiking_neurons else 'rate'
        for test_class in classes:
            test_name = test_class.__name__
            nest.ResetKernel()
            nrn, sim, syn = deepcopy(neuron_params), deepcopy(sim_params), deepcopy(syn_params)
            utils.setup_nest(sim_params, datadir)
            print(f"{test_name} with {spiking_str} neurons:")
            try:
                instance = test_class(nrn, sim, syn, use_spiking_neurons)
            except Exception as e:
                print(f"Test {test_name} raised an exception: ")
                print(e)
                traceback.print_exc()
                print("\n\nProceeding with next test...")
                continue
            else:
                test_passed = instance.evaluate()
                if test_passed:
                    print("\tOK!")
                else:
                    print("\tFAILED!")
                if (not test_passed) or plot_all_runs:
                    filename = os.path.join(imgdir, f"{test_name}_{spiking_str}.png")
                    instance.plot_results()
                    plt.rcParams['savefig.dpi'] = 300
                    plt.savefig(filename)
                    plt.close()
                    print(f"\tTest plot saved under: {filename}")
                print()
