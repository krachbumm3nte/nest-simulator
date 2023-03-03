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
        classes = [eval(arg) for arg in sys.argv[1:]]
    else:
        classes = [FilteredInputCurrent, CurrentConnection, TargetCurrent, DynamicsHX, DynamicsHXMulti, DynamicsHI,
                   DynamicsYH, NetworkDynamics, PlasticityHX, PlasticityHXMulti, PlasticityYH, NetworkPlasticity]

    root, imgdir, datadir = utils.setup_directories(
        "/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit/tests/runs")

    # classes = [NetworkPlasticity]
    sim_params["record_interval"] = 0.1
    sim_params["recording_backend"] = "memory"
    sim_params["datadir"] = datadir
    sim_params["use_mm"] = True

    # increase learning rates to absurd levels to make plasticity visible
    for syn_name in ["ip", "pi", "up"]:
        syn_params["eta"][syn_name] = [10 * lr for lr in syn_params["eta"][syn_name]]

    plot_all_runs = True
    test_results = []

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
                instance.run()
            except Exception as e:
                print(f"Test {test_name} raised an exception: ")
                print(e)
                traceback.print_exc()
                print("\n\nProceeding with next test...")
                continue
            else:
                test_passed = instance.evaluate()
                test_results.append(test_passed)
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

            # remove simulation data from the previous run
            for file in sorted(os.listdir(datadir)):
                f = os.path.join(datadir, file)
                os.remove(f)
    print(f"\n\nAll runs completed.")
    print(f"{sum(test_results)}/{len(test_results)} tests passed.")
