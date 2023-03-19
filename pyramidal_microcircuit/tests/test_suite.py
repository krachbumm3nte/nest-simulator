import sys
import nest
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import traceback
from tests01_neuron_dynamics import *
from tests02_plasticity import *
from itertools import permutations
sys.path.append("/home/johannes/Desktop/nest-simulator/pyramidal_microcircuit")
import utils  # nopep8
from networks.params import Params  # nopep8

if __name__ == "__main__":

    if len(sys.argv) > 1:
        classes = [eval(arg) for arg in sys.argv[1:]]
    else:
        classes = [FilteredInputCurrent, CurrentConnection, TargetCurrent, DynamicsHX, DynamicsHXMulti, DynamicsHI,
                   DynamicsYH, NetworkDynamics, PlasticityHX, PlasticityHXMulti, PlasticityYH, NetworkPlasticity]

    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    root, imgdir, datadir = utils.setup_directories("test_run", root=os.path.join(test_dir, "runs"))


    plot_all_runs = True
    test_results = []

    for use_spiking_neurons, latent_equilibrium in permutations([False, True]):
        params = Params(os.path.join(test_dir, "test_params.json"))
        params.datadir = datadir
        params.spiking = use_spiking_neurons
        params.latent_equilibrium = latent_equilibrium
        
        spiking_str = 'spiking' if use_spiking_neurons else 'rate'
        le_str = '_le' if latent_equilibrium else ""

        params.setup_nest_configs()


        for test_class in classes:
            test_name = test_class.__name__
            nest.ResetKernel()
            utils.setup_nest(params, datadir)
            print(f"{test_name} with {spiking_str} neurons:")
            try:
                instance = test_class(deepcopy(params))
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
            # for file in sorted(os.listdir(datadir)):
            #     f = os.path.join(datadir, file)
            #     os.remove(f)
    print(f"\n\nAll runs completed.")
    print(f"{sum(test_results)}/{len(test_results)} tests passed.")
