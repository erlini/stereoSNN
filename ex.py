import os
import pdb
import spynnaker8 as ps 
import numpy as np
import time
import urllib

class ExternalInputReader():
    def __init__(self, url="",
                 file_path="",
                 crop_xmin=-1,
                 crop_ymin=-1,
                 crop_xmax=-1,
                 crop_ymax=-1,
                 dim_x=1,
                 dim_y=1,
                 sim_time=1000,
                 sim_time_step=0.2,
                 is_rawdata_time_in_ms=False):
        # these are the attributes which contain will contain the sorted, filtered and formatted spikes for each pixel
        self.retinaLeft = []
        self.retinaRight = []

        if url is not "" and file_path is not "" or \
            url is "" and file_path is "":
            print("ERROR: Ambiguous or void input source address. Give either a URL or a local file path.")
            return

        rawdata = None
        eventList = []
        # check the url or the file path, read the data file and pass it further down for processing.
        if url is not "" and url[-4:] == ".dat":
            # connect to website and parse text data
            file = urllib.urlopen(url)
            rawdata = file.read()
            file.close()
            # read the file line by line and extract the timestamp, x and y coordinates, polarity and retina ID.
            # TODO: add a case for different files like the npz, which are read from other servers.
            for e in rawdata.split('\n'):
                if e != '':
                    eventList.append([int(x) for x in e.split()])
        elif url is not "" and url[-4:] == ".npz":
            file = urllib.urlopen(url)
            f = file.read()
            file.close()
            data_left = f["left"]
            data_right = f["right"]
            for t, x, y, p in data_left:
                if is_rawdata_time_in_ms:
                    if t > sim_time:
                        break
                else:
                    if t > sim_time * 1000:
                        break
                eventList.append([float(t), int(x), int(y), int(p), 1])
        elif file_path is not "" and file_path[-4:] == ".dat":
            with open(file_path, 'r') as file:
                rawdata = file.read()
                file.close()
            # read the file line by line and extract the timestamp, x and y coordinates, polarity and retina ID.
            for e in rawdata.split('\n'):
                if e != '':
                    eventList.append([int(x) for x in e.split()])
        elif file_path is not "" and file_path[-4:] == ".npz":
            from numpy import load as load_npz_file
            f = load_npz_file(file_path)
            data_left = f["left"]
            data_right = f["right"]
            for t, x, y, p in data_left:
                if is_rawdata_time_in_ms:
                    if t > sim_time:
                        break
                else:
                    if t > sim_time * 1000:
                        break
                eventList.append([float(t), int(x), int(y), int(p), 1])
            for t, x, y, p in data_right:
                if is_rawdata_time_in_ms:
                    if t > sim_time:
                        break
                else:
                    if t > sim_time * 1000:
                        break
                eventList.append([float(t), int(x), int(y), int(p), 0])

        # initialise the maximum time constant as the total simulation duration. This is needed to set a value
        # for pixels which don't spike at all, since the pyNN frontend requires so.
        # If they spike at the last possible time step, their firing will have no effect on the simulation.
        max_time = sim_time

        # containers for the formatted spikes
        # define as list of lists of lists so that SpikeSourceArrays don't complain
        retinaL = [[[] for y in range(dim_y)] for x in range(dim_x)]
        retinaR = [[[] for y in range(dim_y)] for x in range(dim_x)]

        # last time a spike has occured for a pixel -- used to filter event bursts
        last_tL = [[0.0]]
        last_tR = [[0.0]]
        for x in range(0, dim_x):
            for y in range(0, dim_y):
                last_tL[x].append(-1.0)
                last_tR[x].append(-1.0)
            last_tL.append([])
            last_tR.append([])

        # process each event in the event list and format the time and position. Distribute among the retinas respectively
        for evt in eventList:
            x = evt[1] - 1
            y = evt[2] - 1
            if not is_rawdata_time_in_ms:
                t = evt[0] / 1000.0  # retina event time steps are in micro seconds, so convert to milliseconds
                if t <=sim_time_step:
                    t = sim_time_step+0.01
            else:
                t = evt[0]
                if t <= sim_time_step:
                    t = sim_time_step+0.01

            # firstly, take events only from the within of a window centered at the retina view center,
            # then sort the left and the right events.
            if crop_xmax >= 0 and crop_xmin >= 0 and crop_ymax >= 0 and crop_ymin >= 0:
                if crop_xmin <= x < crop_xmax and crop_ymin <= y < crop_ymax:
                    if evt[4] == 1:
                        # filter event bursts and limit the events to the maximum time for the simulation
                        if t - last_tR[x - crop_xmin][y - crop_ymin] >= 1.0 and t <= max_time:
                            # print "r", x, y, x-lowerBoundX, y-lowerBoundY
                            retinaR[x - crop_xmin][y - crop_ymin].append(t)
                            last_tR[x - crop_xmin][y - crop_ymin] = t
                    elif evt[4] == 0:
                        if t - last_tL[x - crop_xmin][y - crop_ymin] >= 1.0 and t <= max_time:
                            # 				print "l", x, y, x-lowerBoundX, y-lowerBoundY
                            retinaL[x - crop_xmin][y - crop_ymin].append(t)
                            last_tL[x - crop_xmin][y - crop_ymin] = t
            else:
                if evt[4] == 1:
                    # apply the same time filtering
                    if t - last_tR[x][y] >= 1.0 and t <= max_time:
                        retinaR[x][y].append(t)
                        last_tR[x][y] = t
                elif evt[4] == 0:
                    if t - last_tL[x][y] >= 1.0 and t <= max_time:
                        retinaL[x][y].append(t)
                        last_tL[x][y] = t

        # fill the void cells with the last time possible, which has no effect on the simulation. The SpikeSourceArray
        # requires a value for each cell.
        for y in range(0, dim_x):
            for x in range(0, dim_y):
                if retinaR[y][x] == []:
                    retinaR[y][x].append(max_time + 10)
                if retinaL[y][x] == []:
                    retinaL[y][x].append(max_time + 10)

        #minl=[]
        #minr=[]
        #for i in range(0, dim_x):
        #    for j in range(0, dim_y):
        #        minl.append(min(retinaL[i][j]))
        #        minr.append(min(retinaR[i][j]))
        #print min(minl)
        #print min(minr)
        #pdb.set_trace()

        # store the formatted and filtered events which are to be passed to the retina constructors
        self.retinaLeft = retinaL
        self.retinaRight = retinaR


class Retina(object):
    def __init__(self, label="Retina", dimension_x=1, dimension_y=1,
                 use_prerecorded_input=True, spike_times=None,
                 min_disparity=0, experiment_name=None,
                 record_spikes=False, verbose=False):
        assert len(spike_times) >= dimension_x and len(spike_times[0]) >= dimension_y, \
            "ERROR: Dimensionality of retina's spiking times is bad. Retina initialization failed."

        if min_disparity != 0:
            print "WARNING: The minimum disparity is not 0. " \
                  "This may lead to nonsensical results or failure as the network is not tested thoroughly."

        self.label = label
        self.experiment_name = experiment_name
        self.dim_x = dimension_x
        self.dim_y = dimension_y
        self.use_prerecorded_input = use_prerecorded_input

        if verbose:
            print "INFO: Creating Spike Source: {0}".format(label)

        self.pixel_columns = []
        self.labels = []

        if use_prerecorded_input or spike_times is not None:
            for x in range(0, dimension_x - min_disparity):
                retina_label = "{0}_{1}".format(label, x)
                col_of_pixels = ps.Population(dimension_y, ps.SpikeSourceArray(spike_times=spike_times[x]), #{'spike_times': spike_times[x]},
                                               label=retina_label, structure=ps.Line())

                self.pixel_columns.append(col_of_pixels)
                self.labels.append(retina_label)
                if record_spikes:
                    col_of_pixels.record('spikes')

    def get_spikes(self, sort_by_time=True, save_spikes=True):
        print 'get retina spikes'
        spikes_per_population = [x.get_data().segments[0].spiketrains for x in self.pixel_columns]
        #spikes_per_population = [x.get_data('spikes') for x in self.pixel_columns]
        spikes = list()
        for col_index, col in enumerate(spikes_per_population, 0):  # it is 0-indexed
            # for each spike in the population extract the timestamp and x,y coordinates
            x_coord = col_index
            for spike_index, spikestimes in enumerate(col, 0):
                y_coord = spike_index
                for spikestime in spikestimes:
                    spikes.append((round(spikestime, 1), x_coord+1, y_coord+1))
            #for spike in col:
            #    y_coord = int(spike[0])
            #    spikes.append((round(spike[1], 1), x_coord+1, y_coord+1))	# pixel coordinates are 1-indexed
        if sort_by_time:
            spikes.sort(key=lambda x: x[0])
        if save_spikes:
            if not os.path.exists("./spikes"):
                os.makedirs("./spikes")
            i = 0
            while os.path.exists("./spikes/{0}_{1}_spikes_{2}.dat".format(self.experiment_name, i, self.label)):
                i += 1
            with open('./spikes/{0}_{1}_spikes_{2}.dat'.format(self.experiment_name, i, self.label), 'w') as fs:
                fs.write("### DATA FORMAT ###\n"
                        "# Description: These are the spikes a retina has produced (see file name for exact retina label).\n"
                        "# Each row contains: "
                        "Time stamp -- x-coordinate -- y-coordinate\n"
                        "### DATA START ###\n")
                for s in spikes:
                    fs.write(str(s[0]) + " " + str(s[1]) + " " + str(s[2]) + "\n")
                fs.write("### DATA END ###")
                fs.close()
        return spikes


class CooperativeNetwork(object):

    def __init__(self, retinae=None, simulation_time_step=0.1,
                 max_disparity=0, cell_params=None,
                 record_spikes=True, record_v=False, experiment_name="Experiment",
                 verbose=True):
        # IMPORTANT NOTE: This implementation assumes min_disparity = 0

        assert retinae['left'] is not None and retinae['right'] is not None, \
            "ERROR: Retinas are not initialised! Creating Network Failed."

        dx = retinae['left'].dim_x
        assert dx > max_disparity >= 0, "ERROR: Maximum Disparity Constant is illegal!"
        self.max_disparity = max_disparity
        self.min_disparity = 0
        self.size = (2 * (dx - self.min_disparity) * (self.max_disparity - self.min_disparity + 1)
                     - (self.max_disparity - self.min_disparity + 1) ** 2
                     + self.max_disparity - self.min_disparity + 1) / 2
        self.dim_x = dx
        self.dim_y = retinae['left'].dim_y

        # check this assertion before the actual network generation, since the former
        # might take very long to complete.
        assert retinae['left'].dim_x == retinae['right'].dim_x and \
               retinae['left'].dim_y == retinae['right'].dim_y, \
            "ERROR: Left and Right retina dimensions are not matching. Connecting Spike Sources to Network Failed."

        # TODO: make parameter values dependent on the simulation time step
        # (for the case 0.1 it is not tested completely and should serve more like an example)

        # the notation for the synaptic parameters is as follows:
        # B blocker, C collector, S spike source, (2, 4)
        # w weight, d delay, (1)
        # a one's own, z other, (3)
        # i inhibition, e excitation  (5)
        # If B is before C than the connection is from B to C.
        # Example: dSaB would mean a dealy from a spike source to the one's own blocker neuron, and
        # wSzB would be the weight from a spike source to the heterolateral blocker neuron.
        params = {'neural': dict(), 'synaptic': dict(), 'topological': dict()}
        #simulation_time_step = 0.2
        if simulation_time_step == 0.2:
            params['neural'] = {'tau_E': 2.0,
                                'tau_I': 2.0,
                                'tau_mem': 2.07,
                                'v_reset_blocker': -84.0,
                                'v_rest_blocker': -65.0,
                                'v_reset_collector': -90.0,# why -90.0?
                                'v_rest_collector':-65.0} 
            w = 25.0
            params['synaptic'] = {'wBC': w,  # -20.5: negative won't work. However keep in mind that it is inhibitory!
                                  'dBC': simulation_time_step,
                                  'wSC': w,
                                  'dSC': 0.6,
                                  'wSaB': w,
                                  'dSaB': simulation_time_step,
                                  'wSzB': w,    # same story here
                                  'dSzB': simulation_time_step,
                                  'wCCi': w,    # and again
                                  'dCCi': simulation_time_step,
                                  'wCCo': w/3,  # and again
                                  'dCCo': simulation_time_step,
                                  'wCCe': w/10,
                                  'dCCe': simulation_time_step}
            params['topological'] = {'radius_e': 1,
                                     'radius_i': max(self.dim_x, self.dim_y)}
        elif simulation_time_step == 0.1:
            params['neural'] = {'tau_E': 1.0,
                                'tau_I': 1.0,
                                'tau_mem': 1.07,
                                'v_reset_blocker': -92.0,
                                'v_reset_collector': -102.0}
            params['synaptic'] = {'wBC': 39.5, #weight should be positive numbers, indicated as inhibitory synapses (if necessary)!
                                  'dBC': simulation_time_step,
                                  'wSC': 39.5,
                                  'dSC': 0.8,
                                  'wSaB': 49.5,
                                  'dSaB': simulation_time_step,
                                  'wSzB': 39.5, # same here
                                  'dSzB': simulation_time_step,
                                  'wCCi': 50.0, # and here
                                  'dCCi': simulation_time_step,
                                  'wCCe': 4.0,
                                  'dCCe': simulation_time_step}
            params['topological'] = {'radius_e': 1,
                                     'radius_i': max(self.dim_x, self.dim_y)}

        self.cell_params = params if cell_params is None else cell_params
        self.network = self._create_network(record_spikes=record_spikes,
                                            record_v=record_v,
                                            verbose=verbose)
        self._connect_spike_sources(retinae=retinae, verbose=verbose)

        self.experiment_name = experiment_name.replace(" ", "_")

    def _create_network(self, record_spikes=False, record_v=False, verbose=False):

        print("INFO: Creating Cooperative Network of size {0} (in microensembles).".format(self.size))

        network = []
        neural_params = self.cell_params['neural']
        for x in range(0, self.size):
            blocker_columns = ps.Population(self.dim_y * 2,
                                            ps.IF_curr_exp,
                                            {'tau_syn_E': neural_params['tau_E'],
                                             'tau_syn_I': neural_params['tau_I'],
                                             'tau_m': neural_params['tau_mem'],
                                             'v_reset': neural_params['v_reset_blocker'],
                                             'v_rest': neural_params['v_rest_blocker']},
                                             initial_values={'v': neural_params['v_rest_blocker']},
                                            label="Blocker {0}".format(x))

            collector_column = ps.Population(self.dim_y,
                                             ps.IF_curr_exp,
                                             {'tau_syn_E': neural_params['tau_E'],
                                              'tau_syn_I': neural_params['tau_I'],
                                              'tau_m': neural_params['tau_mem'],
                                              'v_reset': neural_params['v_reset_collector'],
                                              'v_rest': neural_params['v_rest_collector']},
                                              initial_values={'v': neural_params['v_rest_collector']},
                                             label="Collector {0}".format(x))

            if record_spikes:
                collector_column.record('spikes')  # records only the spikes
            if record_v:
                collector_column.record_v()  # records the membrane potential -- very resource demanding!
                blocker_columns.record_v()

            network.append((blocker_columns, collector_column))

        self._interconnect_neurons(network, verbose=verbose)
        if self.dim_x > 1:
            self._interconnect_neurons_inhexc(network, verbose)
        else:
            global _retina_proj_l, _retina_proj_r, same_disparity_indices
            _retina_proj_l = [[0]]
            _retina_proj_r = [[0]]
            same_disparity_indices = [[0]]
            
        return network

    def _interconnect_neurons(self, network, verbose=False):

        assert network is not None, \
            "ERROR: Network is not initialised! Interconnecting failed."

        synaptic_params = self.cell_params['synaptic']

        # generate connectivity list: 0 untill dimensionRetinaY-1 for the left
        # and dimensionRetinaY till dimensionRetinaY*2 - 1 for the right
        connList = []
        for y in range(0, self.dim_y):
            connList.append((y, y, synaptic_params['wBC'], synaptic_params['dBC']))
            connList.append((y + self.dim_y, y, synaptic_params['wBC'], synaptic_params['dBC']))

        # connect the inhibitory neurons to the cell output neurons
        if verbose:
            print "INFO: Interconnecting Neurons. This may take a while."
        for ensemble in network:
            ps.Projection(ensemble[0], ensemble[1], ps.FromListConnector(connList), receptor_type='inhibitory')

    def _interconnect_neurons_inhexc(self, network, verbose=False):

        assert network is not None, \
            "ERROR: Network is not initialised! Interconnecting for inhibitory and excitatory patterns failed."

        if verbose and self.cell_params['topological']['radius_i'] < self.dim_x:
            print "WARNING: Bad radius of inhibition. Uniquness constraint cannot be satisfied."
        if verbose and 0 <= self.cell_params['topological']['radius_e'] > self.dim_x:
            print "WARNING: Bad radius of excitation. "

        # create lists with inhibitory along the Retina Right projective line
        nbhoodInhL = []
        nbhoodInhR = []
        nbhoodExcX = []
        nbhoodEcxY = []
        # used for the triangular form of the matrix in order to remain within the square
        if verbose:
            print "INFO: Generating inhibitory and excitatory connectivity patterns."
        # generate rows
        limiter = self.max_disparity - self.min_disparity + 1
        ensembleIndex = 0

        while ensembleIndex < len(network):
            if ensembleIndex / (self.max_disparity - self.min_disparity + 1) > \
                                    (self.dim_x - self.min_disparity) - (self.max_disparity - self.min_disparity) - 1:
                limiter -= 1
                if limiter == 0:
                    break
            nbhoodInhL.append([ensembleIndex + disp for disp in range(0, limiter)])
            ensembleIndex += limiter

        ensembleIndex = len(network)

        # generate columns
        nbhoodInhR = [[x] for x in nbhoodInhL[0]]
        shiftGlob = 0
        for x in nbhoodInhL[1:]:
            shiftGlob += 1
            shift = 0

            for e in x:
                if (shift + 1) % (self.max_disparity - self.min_disparity + 1) == 0:
                    nbhoodInhR.append([e])
                else:
                    nbhoodInhR[shift + shiftGlob].append(e)
                shift += 1

        # generate all diagonals
        for diag in map(None, *nbhoodInhL):
            sublist = []
            for elem in diag:
                if elem is not None:
                    sublist.append(elem)
            nbhoodExcX.append(sublist)

        # generate all y-axis excitation
        for x in range(0, self.dim_y):
            for e in range(1, self.cell_params['topological']['radius_e'] + 1):
                if x + e < self.dim_y:
                    nbhoodEcxY.append(
                        (x, x + e, self.cell_params['synaptic']['wCCe'], self.cell_params['synaptic']['dCCe']))
                if x - e >= 0:
                    nbhoodEcxY.append(
                        (x, x - e, self.cell_params['synaptic']['wCCe'], self.cell_params['synaptic']['dCCe']))

        # Store these lists as global parameters as they can be used to quickly match the spiking collector neuron
        # with the corresponding pixel xy coordinates (same_disparity_indices)
        # TODO: think of a better way to encode pixels: closed form formula would be perfect
        # These are also used when connecting the spike sources to the network! (retina_proj_l, retina_proj_r)

        global _retina_proj_l, _retina_proj_r, same_disparity_indices

        _retina_proj_l = nbhoodInhL
        _retina_proj_r = nbhoodInhR
        same_disparity_indices = nbhoodExcX

        if verbose:
            print "INFO: Connecting neurons for internal excitation and inhibition."

        for row in nbhoodInhL:
            for pop in row:
                for nb in row:
                    if nb != pop:
                        ps.Projection(network[pop][1],
                                      network[nb][1],
                                      #ps.OneToOneConnector(weights=self.cell_params['synaptic']['wCCi'],
                                      #                     delays=self.cell_params['synaptic']['dCCi']),
                                      #target='inhibitory')
                                      ps.OneToOneConnector(),
                                      ps.StaticSynapse(weight=self.cell_params['synaptic']['wCCi'], delay=self.cell_params['synaptic']['dCCi']),
                                      receptor_type='inhibitory')

        for col in nbhoodInhR:
            for pop in col:
                for nb in col:
                    if nb != pop:
                        ps.Projection(network[pop][1],
                                      network[nb][1],
                                      ps.OneToOneConnector(),
                                      ps.StaticSynapse(weight=self.cell_params['synaptic']['wCCi'],
                                                           delay=self.cell_params['synaptic']['dCCi']),
                                      #ps.OneToOneConnector(weights=self.cell_params['synaptic']['wCCi'],
                                      #                     delays=self.cell_params['synaptic']['dCCi']),
                                      #target='inhibitory'
                                      receptor_type='inhibitory')

        for diag in nbhoodExcX:
            for pop in diag:
                for nb in range(1, self.cell_params['topological']['radius_e'] + 1):
                    if diag.index(pop) + nb < len(diag):
                        ps.Projection(network[pop][1],
                                      network[diag[diag.index(pop) + nb]][1],
                                      ps.OneToOneConnector(),
                                      ps.StaticSynapse(weight=self.cell_params['synaptic']['wCCe'],
                                                           delay=self.cell_params['synaptic']['dCCe']),
                                      receptor_type='excitatory')
                                      #ps.OneToOneConnector(weights=self.cell_params['synaptic']['wCCe'],
                                      #                     delays=self.cell_params['synaptic']['dCCe']),
                                      #target='excitatory')
                    if diag.index(pop) - nb >= 0:
                        ps.Projection(network[pop][1],
                                      network[diag[diag.index(pop) - nb]][1],
                                      ps.OneToOneConnector(),
                                      ps.StaticSynapse(weight=self.cell_params['synaptic']['wCCe'],
                                                           delay=self.cell_params['synaptic']['dCCe']),
                                      receptor_type='excitatory')
                                      #ps.OneToOneConnector(weights=self.cell_params['synaptic']['wCCe'],
                                      #                     delays=self.cell_params['synaptic']['dCCe']),
                                      #target='excitatory')

        for ensemble in network:
            ps.Projection(ensemble[1], ensemble[1], ps.FromListConnector(nbhoodEcxY), receptor_type='excitatory')#target='excitatory')

    def _connect_spike_sources(self, retinae=None, verbose=False):

        if verbose:
            print "INFO: Connecting Spike Sources to Network."

        global _retina_proj_l, _retina_proj_r

        # left is 0--dimensionRetinaY-1; right is dimensionRetinaY--dimensionRetinaY*2-1
        connListRetLBlockerL = []
        connListRetLBlockerR = []
        connListRetRBlockerL = []
        connListRetRBlockerR = []
        for y in range(0, self.dim_y):
            connListRetLBlockerL.append((y, y,
                                         self.cell_params['synaptic']['wSaB'],
                                         self.cell_params['synaptic']['dSaB']))
            connListRetLBlockerR.append((y, y + self.dim_y,
                                         self.cell_params['synaptic']['wSzB'],
                                         self.cell_params['synaptic']['dSzB']))
            connListRetRBlockerL.append((y, y,
                                         self.cell_params['synaptic']['wSzB'],
                                         self.cell_params['synaptic']['dSzB']))
            connListRetRBlockerR.append((y, y + self.dim_y,
                                         self.cell_params['synaptic']['wSaB'],
                                         self.cell_params['synaptic']['dSaB']))

        retinaLeft = retinae['left'].pixel_columns
        retinaRight = retinae['right'].pixel_columns
        pixel = 0
        for row in _retina_proj_l:
            for pop in row:
                ps.Projection(retinaLeft[pixel],
                              self.network[pop][1],
                              ps.OneToOneConnector(),
                              ps.StaticSynapse(weight=self.cell_params['synaptic']['wSC'],
                                                   delay=self.cell_params['synaptic']['dSC']),
                              receptor_type='excitatory')
                              #ps.OneToOneConnector(weights=self.cell_params['synaptic']['wSC'],
                              #                     delays=self.cell_params['synaptic']['dSC']),
                              #target='excitatory')
                ps.Projection(retinaLeft[pixel],
                              self.network[pop][0],
                              ps.FromListConnector(connListRetLBlockerL),
                              receptor_type='excitatory')
                              #target='excitatory')
                ps.Projection(retinaLeft[pixel],
                              self.network[pop][0],
                              ps.FromListConnector(connListRetLBlockerR),
                              receptor_type='inhibitory')
                              #target='inhibitory')
            pixel += 1

        pixel = 0
        for col in _retina_proj_r:
            for pop in col:
                ps.Projection(retinaRight[pixel], self.network[pop][1],
                              ps.OneToOneConnector(),
                              ps.StaticSynapse(weight=self.cell_params['synaptic']['wSC'],
                                                   delay=self.cell_params['synaptic']['dSC']),
                              receptor_type='excitatory')
                              #ps.OneToOneConnector(weights=self.cell_params['synaptic']['wSC'],
                              #                     delays=self.cell_params['synaptic']['dSC']),
                              #target='excitatory')
                ps.Projection(retinaRight[pixel],
                              self.network[pop][0],
                              ps.FromListConnector(connListRetRBlockerR),
                              receptor_type='excitatory')
                              #target='excitatory')
                ps.Projection(retinaRight[pixel],
                              self.network[pop][0],
                              ps.FromListConnector(connListRetRBlockerL),
                              receptor_type='inhibitory')
                              #target='inhibitory')
            pixel += 1


    def get_network_dimensions(self):
        parameters = {'size':self.size,
                      'dim_x':self.dim_x,
                      'dim_y':self.dim_y,
                      'min_d':self.min_disparity,
                      'max_d':self.max_disparity}
        return parameters

    """ this method returns (and saves) a full list of spike times
    with the corresponding pixel location and disparities."""
    def get_spikes(self, sort_by_time=True, save_spikes=True):
        print 'get spikes'
        global same_disparity_indices, _retina_proj_l
        #spikes_per_population = [x[1].getSpikes() for x in self.network]
        spikes_per_population = [x[1].get_data().segments[0].spiketrains for x in self.network]
        spikes = list()

        #pdb.set_trace()

        # for each column population in the network, find the x,y coordinates corresponding to the neuron
        # and the disparity. Then write them in the list and sort it by the timestamp value.
        for col_index, col in enumerate(spikes_per_population, 0):  # it is 0-indexed
            # find the disparity
            disp = self.min_disparity
            for d in range(0, self.max_disparity + 1):
                if col_index in same_disparity_indices[d]:
                    disp = d + self.min_disparity
                    break
            # for each spike in the population extract the timestamp and x,y coordinates////////////////
            #for spike in col:
            #    x_coord = 0
            #    for p in range(0, self.dim_x):
            #        if col_index in _retina_proj_l[p]:
            #            x_coord = p
            #            break
            #    y_coord = int(spike[0])
            #    spikes.append((round(spike[1], 1), x_coord+1, y_coord+1, disp))	# pixel coordinates are 1-indexed
            for spike_index, spike_times in enumerate(col, 0):
                for spike_time in spike_times:
                    x_coord = 0
                    for p in range(0, self.dim_x):
                        if col_index in _retina_proj_l[p]:
                            x_coord = p
                            break
                    y_coord = int(spike_index)
                    spikes.append((round(spike_time, 1), x_coord+1, y_coord+1, disp))
        if sort_by_time:
            spikes.sort(key=lambda x: x[0])
        if save_spikes:
            if not os.path.exists("./spikes"):
                os.makedirs("./spikes")
            i = 0
            while os.path.exists("./spikes/{0}_{1}_spikes.dat".format(self.experiment_name, i)):
                i += 1
            with open('./spikes/{0}_{1}_spikes.dat'.format(self.experiment_name, i), 'w') as fs:
                self._write_preamble(fs)
                fs.write("### DATA FORMAT ###\n"
                        "# Description: All spikes from the Collector Neurons are recorded. The disparity is inferred "
                        "from the Neuron ID. The disparity is calculated with the left camera as reference."
                        "The timestamp is dependent on the simulation parameters (simulation timestep).\n"
                        "# Each row contains: "
                        "Time -- x-coordinate -- y-coordinate -- disparity\n"
                        "### DATA START ###\n")
                for s in spikes:
                    fs.write(str(s[0]) + " " + str(s[1]) + " " + str(s[2]) + " " + str(s[3]) + "\n")
                fs.write("### DATA END ###")
                fs.close()
        return spikes

    """ this method returns the accumulated spikes for each disparity as a list. It is not very useful except when
    the disparity sorting and formatting in the more general one get_spikes is not needed."""
    def get_accumulated_disparities(self, sort_by_disparity=True, save_spikes=True):
        if sort_by_disparity:
            global same_disparity_indices
            spikes_per_disparity_map = []
            for d in range(0, self.max_disparity - self.min_disparity + 1):
                collector_cells = [self.network[x][1] for x in same_disparity_indices[d]]
                spikes_per_disparity_map.append(sum([sum(x.get_spike_counts().values()) for x in collector_cells]))
                if save_spikes:
                    if not os.path.exists("./spikes"):
                        os.makedirs("./spikes")
                    i = 0
                    while os.path.exists("./spikes/{0}_{1}_disp.dat".format(self.experiment_name, i)):
                        i += 1
                    with open('./spikes/{0}_{1}_disp.dat'.format(self.experiment_name, i), 'w') as fd:
                        self._write_preamble(fd)
                        for s in spikes_per_disparity_map:
                            fd.write(str(s) + "\n")
                        fd.close()
                return spikes_per_disparity_map
        else:
            # this is pretty useless. maybe it should be removed in the future
            all_spikes = sum(sum(x[1].get_spikes_count().values() for x in self.network))
            return all_spikes

    """ this method returns a list containing the membrane potential of all neural populations sorted by id."""
    def get_v(self, save_v=True):
        voltages = {"collector_v": [x[1].get_v() for x in self.network],
                    "blockers_v":[x[0].get_v() for x in self.network]}
        if save_v:
            if not os.path.exists("./membrane_potentials"):
                os.makedirs("./membrane_potentials")
            i = 0
            while os.path.exists("./membrane_potentials/{0}_{1}_vmem.dat".format(self.experiment_name, i)):
                i += 1
            with open('./membrane_potentials/{0}_{1}_vmem.dat'.format(self.experiment_name, i), 'w') as fv:
                self._write_preamble(fv)
                fv.write("### DATA FORMAT ###\n"
                        "# Description: First all Blocker Populations are being printed. "
                        "Then all Collector populations. Both are sorted by Population ID (i.e. order of creation). "
                        "Each Blocker/Collector Population lists all neurons, sorted by Neuron ID. "
                        "There are two times more Blocker than Collector Neurons.\n"
                        "# Each row contains: "
                        "Blocker/Collector tag (b/c) -- Population ID -- Neuron ID -- Time -- Membrane Potential\n"
                        "### DATA START ###\n")
                for pop_id, pop_v in enumerate(voltages["blockers_v"]):
                    for v in pop_v:
                        fv.write("b " + str(int(pop_id)) + " " + str(int(v[0])) + " " + str(v[1]) + " " + str(v[2]) + "\n")
                for pop_id, pop_v in enumerate(voltages["collector_v"]):
                    for v in pop_v:
                        fv.write("c " + str(int(pop_id)) + " " + str(int(v[0])) + " " + str(v[1]) + " " + str(v[2]) + "\n")
                fv.write("### DATA END ###")
                fv.close()
        return voltages

    def _write_preamble(self, opened_file_descriptor):
        if opened_file_descriptor is not None:
            f = opened_file_descriptor
            f.write("### PREAMBLE START ###\n")
            f.write("# Experiment name: \n\t{0}\n".format(self.experiment_name))
            f.write("# Network parameters "
                    "(size in ensembles, x-dimension, y-dimension, minimum disparity, maximum disparity, "
                    "radius of excitation, radius of inhibition): "
                    "\n\t{0} {1} {2} {3} {4} {5} {6}\n".format(self.size, self.dim_x, self.dim_y,
                                                         self.min_disparity, self.max_disparity,
                                                         self.cell_params['topological']['radius_e'],
                                                         self.cell_params['topological']['radius_i']))
            f.write("# Neural parameters "
                    "(tau_excitation, tau_inhibition, tau_membrane, v_reset_blocker, v_reset_collector): "
                    "\n\t{0} {1} {2} {3} {4}\n".format(self.cell_params['neural']['tau_E'],
                                                 self.cell_params['neural']['tau_I'],
                                                 self.cell_params['neural']['tau_mem'],
                                                 self.cell_params['neural']['v_reset_blocker'],
                                                 self.cell_params['neural']['v_reset_collector']))
            f.write('# Synaptic parameters '
                    '(wBC, dBC, wSC, dSC, wSaB, dSaB, wSzB, dSzB, wCCi, dCCi, wCCe, dCCe): '
                    '\n\t{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'
                    .format(self.cell_params['synaptic']['wBC'],
                            self.cell_params['synaptic']['dBC'],
                            self.cell_params['synaptic']['wSC'],
                            self.cell_params['synaptic']['dSC'],
                            self.cell_params['synaptic']['wSaB'],
                            self.cell_params['synaptic']['dSaB'],
                            self.cell_params['synaptic']['wSzB'],
                            self.cell_params['synaptic']['dSzB'],
                            self.cell_params['synaptic']['wCCi'],
                            self.cell_params['synaptic']['dCCi'],
                            self.cell_params['synaptic']['wCCe'],
                            self.cell_params['synaptic']['dCCe']))
            f.write('# Comments: Caution: The synaptic parameters may vary according with '
                    'different simulation time steps. To understand the abbreviations for the '
                    'synaptic parameters, see the code documentation.\n')
            f.write("### PREAMBLE END ###\n")


class SNNSimulation(object):
    def __init__(self, simulation_time=1000, simulation_time_step=0.2, threads_count=8):
        self.simulation_time = simulation_time
        self.time_step = simulation_time_step
        # setup timestep of simulation and minimum and maximum synaptic delays
        ps.setup(timestep=simulation_time_step,
                 min_delay=simulation_time_step,
                 max_delay=10*simulation_time_step,
                 #n_chips_required = n_chips_required,
                 threads=threads_count)

    def run(self):
        # run simulation for time in milliseconds
        ps.run(self.simulation_time)

    def end(self):
        # finalise program and simulation
        ps.end()


def run_experiment_pendulum(with_visualization=False):
    """
    TODO: add experiment description.

    """
    experiment_name = "Pendulum30"
    experiment_duration = 1000  # in ms
    time_step = 0.2
    dx = 72  # in pixels
    dy = 84  # in pixels
    #max_d = 42  # in pixels
    max_d = 10  # in pixels
    crop_xmin = 32  # in pixels
    crop_ymin = 22  # in pixels

    # Setup the simulation
    Simulation = SNNSimulation(simulation_time=experiment_duration,simulation_time_step=time_step, threads_count=8)

    # Define the input source
    path_to_input = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "./data/input/pendulum_left_30cm_2.tsv.npz")
    ExternalRetinaInput = ExternalInputReader(file_path=path_to_input,
                                              dim_x=dx,
                                              dim_y=dy,
                                              crop_xmin=crop_xmin,
                                              crop_xmax=crop_xmin + dx,
                                              crop_ymin=crop_ymin,
                                              crop_ymax=crop_ymin + dy,
                                              sim_time=experiment_duration,
                                              sim_time_step=time_step,
                                              is_rawdata_time_in_ms=False)
    # Create two instances of Retinas with the respective inputs
    RetinaL = Retina(label="RetL", dimension_x=dx, dimension_y=dy,
                     spike_times=ExternalRetinaInput.retinaLeft,
                     record_spikes=True,
                     experiment_name=experiment_name)
    RetinaR = Retina(label="RetR", dimension_x=dx, dimension_y=dy,
                     spike_times=ExternalRetinaInput.retinaRight,
                     record_spikes=True,
                     experiment_name=experiment_name)

    # Create a cooperative network for stereo vision from retinal disparity
    SNN_Network = CooperativeNetwork(retinae={'left': RetinaL, 'right': RetinaR},
                                     simulation_time_step=time_step,
                                     max_disparity=max_d,
                                     record_spikes=True,
                                     record_v=False,
                                     experiment_name=experiment_name)

    # Start the simulation
    Simulation.run()

    RetinaL.get_spikes()
    RetinaR.get_spikes()

    # Store the results in a file
    SNN_Network.get_spikes(sort_by_time=True, save_spikes=True)

    # Finish the simulation
    Simulation.end()

    if with_visualization:
        from visualizer import Visualizer
        network_dimensions = SNN_Network.get_network_dimensions()
        #     network_dimensions = {'dim_x':dx, 'dim_y':dy, 'min_d':0, 'max_d':max_d}
        viz = Visualizer(network_dimensions=network_dimensions,
                         experiment_name=experiment_name,
                         spikes_file="./spikes/Pendulum30_0_spikes.dat")
        # viz.microensemble_voltage_plot(save_figure=True)
        viz.disparity_histogram(over_time=True, save_figure=True)
        # viz.scatter_animation(dimension=3, save_animation=True, rotate=True)

run_experiment_pendulum()
