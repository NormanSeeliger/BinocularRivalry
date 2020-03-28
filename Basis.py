"""
@Topic: Main concept for master's thesis, layer IV and II/III SORNS

@author: Norman Seeliger
@date: 20.03.2017
@email: seeliger.norman@gmail.com
"""

#from brian import *
from brian2 import * 
import neuronparameters as pms
import math
import Input
import random as rn
import numpy as np
from operator import sub, add
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d import Axes3D
import time
# FFIM

alpha_val = pms.alpha_val   # float(sys.argv[1]) # alpha(LTD) constant for iSTDP. for example, 0.05.
eta_i_LTP = 0.18            # float(sys.argv[2]) # iSTDP window LTP amplitude (LTP learning rate). for example, 1.
ampl_eTOi = -0.05           # amplitude fpr exc. to inh. STDP windows

vunit = 0.001               # transfer of values
run_time = 30000 * ms
stop =    30000               # unitless time, some functions otherweise report problems

############################################ NEURON PARAMETERS #######################################################
sparse_eTOe = pms.sparse_eTOe       # sparseness e to e layer 4
sparse_iTOe = pms.sparse_iTOe       # sparseness i to e layer 4
sparse_eTOi = pms.sparse_eTOi       # sparseness e to i layer 4
sparse_iTOi = pms.sparse_iTOi       # sparseness i to i layer 4
N_e = pms.N_e                       # amount of exc neurons layer 4
N_i = pms.N_i                       # amount of inh neurons layer 4
N_th = pms.N_th                     # amount of thalamic input neurons
N_Bg = pms.N_Bg                     # amount of background neurons providing noise
wi_eTOe = pms.wi_eTOe               # baseline weight, modified for AMPA and NMDA currents
wi_eTOi = pms.wi_eTOi
wi_iTOe = pms.wi_iTOe
wi_iTOi = pms.wi_iTOi
gapweight = pms.gapweight # EPSC/IPSC weight constant for gap junctons
delay_eTOe = pms.delay_eTOe * ms    # cortical delays, taken from layer 5 (Daniel Miner's Code)
delay_eTOi = pms.delay_eTOi * ms
delay_iTOe = pms.delay_iTOe * ms
delay_iTOi = pms.delay_iTOi * ms

sigma_noise = pms.noise_level * mV  # noise amplitude
tau = pms.taum * ms                 # membrane time constant
Vr_e = pms.Vr_e * mV                # excitatory reset value
Vr_i = pms.Vr_i * mV                # inhibitory reset value
El = pms.El * mV                    # resting value
Vti = pms.Vti * mV                  # minus maximum initial threshold voltage
Vtvar = pms.Vtvar * mV              # maximum initial threshold voltage swing
Vvi = pms.Vvi * mV                  # minus maximum initial voltage
Vvar = pms.Vvar * mV                # maximum initial voltage swing

Ee = pms.Ee * mV                    # reversal potential exc current (AMPA AND NMDA)
Ei = pms.Ei * mV                    # "" inhibitory neurons
Eib = pms.Eib * mV
Ek = pms.E_k * mV                   # reversal potential bAHP current (or M-Current, the latter is not implemented yet)

tau_e = pms.taue * ms               # time constant AMPRA currents (all neuros)
tau_nmda = pms.tau_nmda * ms        # time constant NMDA currents
tau_i = pms.taui * ms               # time constant GABA_A currents
taub1 = pms.taub1 * ms              # time constant GABA_B (first)
taub2 = pms.taub2 * ms              # time constant GABA_B (second)

### STDP parameters
taupre = pms.taupre * ms            # pre-before-post STDP time constant
taupost = pms.taupost * ms          # post-before-pre STDP time constant
Ap = pms.Ap                         # potentiating STDP learning rate
Ad = pms.Ad                         # depressing STDP learning rate

gmax_iTOe = 20* vunit
### Synaptic normalization parameters/maximum weights
total_in_eTOe = pms.total_in_eTOe
total_in_iTOe = pms.total_in_iTOe
total_in_eTOi = pms.total_in_eTOi
total_in_iTOi = pms.total_in_iTOi

### iSTDP parameters
alpha = alpha_val * vunit           # LTD factor for iSTDP
ampl = eta_i_LTP * vunit            # LTP amplitude for iSTDP
tauprei = pms.tauprei * ms          # time constant of iSTDP window
wmaxi = total_in_iTOe
tau_sra = pms.tau_sra * ms
g_sra = pms.g_sra
tau_mcurr = pms.tau_mcurr * ms

# parameters
orientations = 4                    # 45 degree change
orientation_populations = list()
sigma_ee = 3                        # gaussian connectivity profile constants
sigma_ee_2from4 = 4.5
sigma_ee_2to2 = 3.5
sigma_ei = 20
sigma_ie = 2
sigma_ii = 2
sigma_ii_gap = 2
sparseness_ind = pms.sparse_eTOe    # sparseneess for layer 4 to layer 2/3 connectivities
sparseness_inh = pms.sparse_iTOe
input_strength = 4
#E_1 = 1

# running options
SN_ON = True
STDP_ON = False
STP_ON = False
SRA_ON = True
Exercise = False
Patching = True
Amblyopia = False
GAPS_ON = False
Record_weights = False
simple_weights = True # ust normal weight evolutions 
Kesten = False # Kesten process for synaptic weights
input_mode = 'complex' # corr_equal - corr_sep - poisson - complex - inc_contrast

# clocks
slow_steps = 5000 * ms
#slow_clock = Clock(slow_steps,makedefaultclock=False)
fast_steps = 100 * ms
#fast_clock = Clock(fast_steps,makedefaultclock= False
middle_steps = 1000 * ms# )
exercise_onset = run_time / 3

################################### DEFINING ORIENTATION PREFERENCE DEPENDENT ON LOCATION IN NETWORK #################
def row_Ex(x):
    """Returns the row of the excitatory neuron"""
    return (x - x % (2 * math.sqrt(N_e / 2))) / (2 * math.sqrt(N_e / 2))

def column_Ex(x):
    """Return the column of a excitatory neuron"""
    return (x % (2 * math.sqrt(N_e / 2)))

def row_In(x):
    """Returns the row of the inhibitory neuron"""
    return 2 * (x - x % (2 * math.sqrt(N_i / 2))) / (2 * math.sqrt(N_i / 2)) + 1

def column_In(x):
    """Return the column of a inhibitory neuron"""
    return 2 * (x % (2 * math.sqrt(N_i / 2))) + 1

def orientation_Ex(x):
    """Assigning an orientation dependent on position within each ODC"""
    if column_Ex(x) >= math.sqrt(N_e / 2):  # right eye
        if row_Ex(x) < math.sqrt(N_e / 2) / 2:  # upper half
            if column_Ex(x) <= row_Ex(x) + math.sqrt(N_e / 2):
                return 45
            elif column_Ex(x) >= (math.sqrt(N_e / 2) - row_Ex(x)) + math.sqrt(N_e / 2):
                return 135
            else:
                return 0
        else:  # lower half
            if column_Ex(x) < (math.sqrt(N_e / 2) - row_Ex(x)) + math.sqrt(N_e / 2) - 1:
                return 45
            elif column_Ex(x) >= row_Ex(x) + math.sqrt(N_e / 2):
                return 135
            else:
                return 90
    else:
        if row_Ex(x) < math.sqrt(N_e / 2) / 2:  # upper half
            if column_Ex(x) < row_Ex(x):
                return 135
            elif column_Ex(x) >= (math.sqrt(N_e / 2) - row_Ex(x) - 1):
                return 45
            else:
                return 0
        else:  # lower half
            if column_Ex(x) < (math.sqrt(N_e / 2) - row_Ex(x)):
                return 135
            elif column_Ex(x) > row_Ex(x):
                return 45
            else:
                return 90

def orientation_In(x):
    """Assigning an orientation dependent on position within each ODC"""
    if column_In(x) >= math.sqrt(N_e / 2):  # right eye
        if row_In(x) < math.sqrt(N_e / 2) / 2:  # upper half
            if column_In(x) <= row_In(x) + math.sqrt(N_e / 2):
                return 45
            elif column_In(x) > (math.sqrt(N_e / 2) - row_In(x)) + math.sqrt(N_e / 2):
                return 135
            else:
                return 0
        else:  # lower half
            if column_In(x) < (math.sqrt(N_e / 2) - row_In(x)) + math.sqrt(N_e / 2) - 1:
                return 45
            elif column_In(x) >= row_In(x) + math.sqrt(N_e / 2):
                return 135
            else:
                return 90
    else:
        if row_In(x) < math.sqrt(N_e / 2) / 2:  # upper half
            if column_In(x) < row_In(x):
                return 135
            elif column_In(x) >= (math.sqrt(N_e / 2) - row_In(x) - 1):
                return 45
            else:
                return 0
        else:  # lower half
            if column_In(x) <= (math.sqrt(N_e / 2) - row_In(x)):
                return 135
            elif column_In(x) > row_In(x):
                return 45
            else:
                return 90


def distance_ee(x, y):
    """Returns the distance between two neurons within the 2D architecture"""
    return math.sqrt((row_Ex(x) - row_Ex(y)) ** 2 + (column_Ex(x) - column_Ex(y)) ** 2)

def distance_ie(x, y):
    """Returns the distance between two neurons within the 2D architecture"""
    return max(1, math.sqrt((row_In(x) - row_Ex(y)) ** 2 + (column_In(x) - column_Ex(y)) ** 2))

def distance_ei(x, y):
    """Returns the distance between two neurons within the 2D architecture"""
    return max(1, math.sqrt((row_Ex(x) - row_In(y)) ** 2 + (column_Ex(x) - column_In(y)) ** 2))

def islefteye(x):
    """Returns 1 if the neuron is driven by the left eye, 0 otherwise"""
    return (column_Ex(x) < math.sqrt(N_e / 2))

def islefteye_In(x):
    """Returns 1 if the neuron is driven by the left eye, 0 otherwise"""
    return (column_In(x) < math.sqrt(N_e / 2))


################################### NEURON GROUP INITIALIZATION ######################################################
eqs = Equations('''
      dV / dt =  (El - V + ge*(Ee-V) + gi*(Ei-V)+ gib*(Eib-V) + gsra*(Ek-V) + gmcurr*(Ek-V)+ (geNMDA*(Ee-V)/(mg_dep)))/tau + sigma_noise * xi / (tau **.5) : volt
      Vt : volt
      dge/dt = -ge/tau_e            : 1
      dgi/dt = -gi/tau_i            : 1
      dgib/dt = - gib/taub1 - gib/taub2 : 1
      dgsra/dt = -gsra/tau_sra      : 1
      dgeNMDA/dt = -geNMDA/tau_nmda  : 1
      dgmcurr/dt = -gmcurr/tau_mcurr : 1
      mg_dep = 1 + exp(-0.062 * V/volt)/3.57 : 1
        ''')  # + sigma_noise * xi / (tau **.5)

# excitatory neurons
G_e = NeuronGroup(N_e, model=eqs, threshold='V > Vt', reset='''V = Vr_e 
                                                                 gsra += g_sra''' )
#G_e23 = NeuronGroup(pms.N_e_23, model=eqs, threshold='V > Vt', reset=Vr_e)
#G_Layer2 = NeuronGroup(1, model=eqs, threshold='V > Vt', reset=Vr_e)
# defining of subgroups for later compuation of population behavior
G_e000l = [x for x in range(N_e) if orientation_Ex(x) == 0 and x % 20 < 10]
G_e000r = [x for x in range(N_e) if orientation_Ex(x) == 0 and x % 20 >= 10]
G_e045l = [x for x in range(N_e) if orientation_Ex(x) == 45 and x % 20 < 10];
G_e045r = [x for x in range(N_e) if orientation_Ex(x) == 45 and x % 20 >= 10];
G_e090l = [x for x in range(N_e) if orientation_Ex(x) == 90 and x % 20 < 10];
G_e090r = [x for x in range(N_e) if orientation_Ex(x) == 90 and x % 20 >= 10];
G_e135l = [x for x in range(N_e) if orientation_Ex(x) == 135 and x % 20 < 10];
G_e135r = [x for x in range(N_e) if orientation_Ex(x) == 135 and x % 20 >= 10];

# inhibitory neurons, same as above but different channels and currents
eqs_in = Equations('''
      dV / dt =  (Igap + El - V + ge*(Ee-V) + gi*(Ei-V)+ gsra*(Ek-V)+(geNMDA*(Ee-V)/(mg_dep)))/tau  + sigma_noise * xi / (tau **.5) : volt
      Vt : volt
      dge/dt = -ge/tau_e            : 1
      dgi/dt = -gi/tau_i            : 1
      dgsra/dt = -gsra/tau_sra      : 1

      Igap : volt
      dgeNMDA/dt = -geNMDA/tau_nmda    : 1
      mg_dep = 1 + exp(-0.062 * V/volt)/3.57 : 1
        ''')

G_i = NeuronGroup(N_i, model=eqs_in, threshold='V > Vt', reset='''V = Vr_i''')
G_i23 = NeuronGroup(pms.N_i_23, model=eqs_in, threshold='V > Vt', reset='''V = Vr_e''')
G_i000l = [x for x in range(N_i) if orientation_In(x) == 0 and x % 10 < 5]
G_i000r = [x for x in range(N_i) if orientation_In(x) == 0 and x % 10 >= 5]
G_i045l = [x for x in range(N_i) if orientation_In(x) == 45 and x % 10 < 5];
G_i045r = [x for x in range(N_i) if orientation_In(x) == 45 and x % 10 >= 5];
G_i090l = [x for x in range(N_i) if orientation_In(x) == 90 and x % 10 < 5];
G_i090r = [x for x in range(N_i) if orientation_In(x) == 90 and x % 10 >= 5];
G_i135l = [x for x in range(N_i) if orientation_In(x) == 135 and x % 10 < 5];
G_i135r = [x for x in range(N_i) if orientation_In(x) == 135 and x % 10 >= 5];

### Randomize initial voltages
G_e.V = -(Vvi + rand(N_e) * Vvar)  # membrane potentials (initial)
G_i.V = -(Vvi + rand(N_i) * Vvar)  # membrane potentials (initial)
#G_Layer2.V = -(Vvi + 0.5 * Vvar)

# Neuronal spiking thresholds (initial)
#G_e.Vt = pms.V_Thresh_e * mV
G_e.Vt = np.random.normal(pms.V_Thresh_e,0.1,N_e) * mV
G_i.Vt = np.random.normal(pms.V_Thresh_i,0.1,N_i) * mV
#G_Layer2.Vt = pms.V_Thresh_i * mV



##########################################  CONNECTIONS LAYER 4 #######################################################
# excitatory E->E connections layer 4
taud=600*ms
tauf=21*ms
U=0.53
C_ee = Synapses(G_e, G_e, ''' w : 1
                dx/dt = (1-x)/taud  : 1 (clock-driven)
                du/dt = (U-u)/tauf  : 1 (clock-driven)
                ''',
                on_pre = ''' ge += w
                x = x * (1-u)
                u = u + U*(1-u)
                w = (u*x) * w            
                ''')
C_ee.connect()
C_ee.delay = delay_eTOe
for i in range(N_e):
    counter = 0
    while counter < pms.sparse_eTOe * N_e:
        # Pick random neuron, if within same OCD (alternatively, allow long range connections between similar orientations
        # as long-range projections), then assign a probability for a connection and depending on that,
        # decide whether to connect or not (not to neurons preferring the opposite stimulus)
        j = rn.randint(0, N_e - 1)
        dist_effect = ((1.0 / math.sqrt(2 * math.pi * (sigma_ee ** 2))) * \
                       math.exp(-((distance_ee(i, j)) ** 2 / (2 * (sigma_ee ** 2))))) / (
                      3 * abs(islefteye(i) - islefteye(j)) + 1)
        if rn.random() < dist_effect and i != j and C_ee.w[i,j][0] == 0.0 and \
                        abs(orientation_Ex(i) - orientation_Ex(j)) != 90 and \
                        abs(islefteye(i) - islefteye(j)) == 0:  # no autopses
            if (orientation_Ex(i) - orientation_Ex(j) == 0):
                C_ee.connect(i = i, j = j)
                C_ee.w[i,j] = 0.25 * wi_eTOe
                #C_ee.delay[i,j] = 0.75 * delay_eTOe  + 0.5 * rn.random()*delay_eTOe*distance_ee(i,j)
            else:
                C_ee.connect(i = i, j = j)
                C_ee.w[i,j] = 0.05 * wi_eTOe
                #C_ee.delay[i, j] = 0.75 * delay_eTOe + 0.5 * rn.random() * delay_eTOe * distance_ee(i, j)
                #counter = counter - 1
            counter += 1
            
# AMPFA-currents with 0.25 of baseline, NMDA-Currents with 0.75 of baseline for every connection where an AMPA-receptor
# has been established
C_ee_NMDA = Synapses(G_e,G_e,'w : 1','geNMDA += w',delay=delay_eTOe)
C_ee_NMDA.connect()
for i in range(N_e) :
    for j in range(N_e) :
        if C_ee.w[i,j][0] != 0.0 and C_ee_NMDA.w[i,j][0] == 0.0 :
            C_ee_NMDA.connect(i = i, j= j)
            C_ee_NMDA.w[i,j] = 3 * C_ee.w[i,j]
           #C_ee_NMDA.delay[i, j] = 0.75 * delay_eTOe + 0.5 * rn.random() * delay_eTOe * distance_ee(i, j)


# excitatory E->I connections Layer 4
C_ei = Synapses(G_e,G_i,'w : 1','ge += w',delay= delay_eTOi)# delay_eTOi)
C_ei.connect()
for i in range(N_e) :
    counter = 0
    while counter < pms.sparse_eTOi *N_i :
        # Pick random inhibitory neuron with a bias towards the other eye (alternatively only the other eye),
        # assign connection probability and establish connections
        j = rn.randint(0,N_i-1)
        dist_effect  = ((1.0 / math.sqrt(2 * math.pi * (sigma_ei ** 2))) * \
                       math.exp(-((distance_ei(i,j)) ** 2 / (2 * (sigma_ei ** 2))))) \
                       * (3*abs(islefteye(i)-islefteye_In(j))+1)
        # facilitating projections towards inhibitory neurons of the other eye
        if rn.random() < dist_effect and C_ei.w[i,j][0] == 0 and abs(islefteye(i)-islefteye_In(j)) == 1 \
                and (orientation_Ex(i)-orientation_In(j) != 0):
            if abs(orientation_Ex(i)-orientation_In(j)) == 90 :
                C_ei.connect(i = i, j = j)
                C_ei.w[i,j] = 0.25 * wi_eTOi
            else :
                C_ei.connect(i = i, j = j)
                C_ei.w[i,j] = 0.25 * wi_eTOi
            counter += 1
            #C_ei.delay[i,j] = 0.75 * delay_eTOi + 0.5 * rn.random() * delay_eTOi * distance_ei(i, j)

# NNDA receptors onto inhibitory neurons, same reasoning as for excitatory connections
C_ei_NMDA = Synapses(G_e,G_i,'w : 1','geNMDA += w',delay= delay_eTOi)
C_ei_NMDA.connect()
for i in range(N_e) :
    for j in range(N_i):
        if C_ei.w[i, j][0] != 0.0 and C_ei_NMDA.w[i, j][0] == 0.0:
            C_ei_NMDA.connect(i=i, j=j)
            C_ei_NMDA.w[i, j] = 3 * C_ei.w[i,j]
            #C_ei_NMDA.delay[i, j] = 0.75 * delay_eTOi + 0.5 * rn.random() * delay_eTOi * distance_ei(i, j)

# inihibtory I->I connections layer 4 - without further details, random connections
C_ii = Synapses(G_i,G_i,'w : 1','gi += w',delay= delay_iTOi)
C_ii.connect(p = pms.sparse_iTOi)
C_ii.w = weight=wi_iTOi

C_iigap = Synapses(G_i,G_i,model = '''w : 1
                                  Igap_post = w* (V_pre - V_post) : volt (summed)''')
#C_iigap.connect()
for m in range(N_i) :
    for n in range(N_i) :
        if (abs(row_In(m)-row_In(n)) <= 2.0 and abs(column_In(m) -column_In(n)) <= 2.0):
            C_iigap.connect(i = m,j = n)
            C_iigap.connect(i = n, j = m)

C_iigap.w = gapweight
#G_i.Igap = C_iigap.Igap_post

# Gap junctions for PV+ neurons, seemm to be important and form specific subnetworks with a certain connectivity
# not implemented yet
# S=Synapses(G_i,model='''w:1 # gap junction conductance
#                             Igap=w*(V_pre-V_post): 1''')
# for i in range(N_i) :
#     for j in range(N_i) :
#         if (math.sqrt((row_In(i) - row_In(j)) ** 2 + (column_In(i) - column_In(j)) ** 2) < sigma_ii_gap) :
#             S[i,j] = True
#             S[j,i] = True # reciprocal connection for proximal connections
# G_i.Igap=S.Igap
# S.w=.02
    
# inhibitory I-> E connections layer 4

C_ie = Synapses(G_i,G_e,
                '''w : 1
                dApi/dt=-Api/tauprei : 1 (event-driven)
                dApi2/dt=-Api2/tauprei : 1 (event-driven)''',
                on_pre = '''gi += w
                Api += ampl
                w+= Api2-alpha''',
                on_post = '''Api2 +=ampl
                w+=Api ''' )

already_used = list()
used_matrix = list()
for m in range(N_i) :
    already_used = list()
    counter = 0
    while counter  < pms.sparse_iTOe *N_e : #  C_ie.N_outgoing[m] 
        # inhibitory neurons target neeurons with opposite preferences and are driven by a preference opposite to
        # their taget's preference - e.g. inh_90 is driven by exc_0 and inhibitirs exc_90, therefore only the naiming
        # is different
        # Bias towards the same eye, however there are inhibitory connections spanning ODC borders
        n = rn.randint(0,N_e-1)
        dist_effect  = ((1.0 / math.sqrt(2 * math.pi * (sigma_ie ** 2))) * \
                       math.exp(-((distance_ie(m,n)) ** 2 / (2 * (sigma_ie ** 2))))) / \
                       (3*abs(islefteye_In(m)-islefteye(n))+1)
        # facilitates projections towards excitator neurons of the same eye
        if (rn.random() < dist_effect and n not in already_used):
            if islefteye_In(m) - islefteye(n) == 0 : # same eye, 1-1 or 0-0 :
                already_used.append(n)
                C_ie.connect(i = m, j = n)
                if orientation_In(m) == 90 and orientation_Ex(n) == 90  :
                    C_ie.w[m, n] = 1 * wi_iTOe
                elif abs(orientation_In(m) - orientation_Ex(n)) == 0 :
                    C_ie.w[m, n] = 1 * wi_iTOe
                elif abs(orientation_In(m) - orientation_Ex(n)) == 45 \
                        or abs(orientation_In(m) - orientation_Ex(n)) == 135 :
                    C_ie.w[m, n] = 1 * wi_iTOe
                else :
                    counter = counter - 1
            else :
                if orientation_In(m) - orientation_Ex(n) == 0 :
                    C_ie.connect(i = m, j = n)
                    C_ie.w[m, n] = 1 * wi_iTOe
                    already_used.append(n)
                elif abs(orientation_In(m) - orientation_Ex(n)) == 45 or \
                                abs(orientation_In(m) - orientation_Ex(n)) == 135 :
                    C_ie.connect(i = m, j = n)
                    C_ie.w[m, n] =  1* wi_iTOe
                    already_used.append(n)
                else :
                    counter = counter - 1
            #     # TODO: This aspect is curcial, so main target for changes
            counter += 1
            C_ie.delay[m, n] = 0.75 * delay_iTOe + 0.5 * rn.random() * delay_iTOe #* distance_ie(i, j)
            if counter == pms.sparse_iTOe *N_e :
                used_matrix.append(already_used)



#C_ie_GABAB = Synapses(G_i,G_e,'''w : 1''',
#                 on_pre = '''gib += w ''',delay=pms.delay_GABAB * ms)


 
# =============================================================================
# C_ie_GABAB = Synapses(G_i,G_e,'''w : 1
#     dApi/dt=-Api/tauprei : 1 (event-driven)
#     dApi2/dt=-Api2/tauprei : 1 (event-driven)''',
#     on_pre = '''gib += w
#     Api += ampl
#     w+= Api2-alpha''',
#     on_post = '''Api2 +=ampl
#     w+=Api ''',delay=pms.delay_GABAB * ms)
# 
# 
# 
# C_ie_GABAB.connect()
# for i in range(N_i) :
#     for j in range(N_e) :
#         if j in used_matrix[i]  :
#             C_ie_GABAB.connect(i = i, j = j)
#             C_ie_GABAB.w[i,j] = 0.2 * C_ie.w[i,j][0]
# =============================================================================



print('Connections within layer 4 done.')

########################################## TOWARDS LAYER 23 #####################################################
# currently not implemented

#C_ee42 = Connection(G_e,G_Layer2,'ge',delay=pms.delay_4TO2,weight=wi_eTOe,sparseness = 1.0)
# C_ee42 = Connection(G_e,G_e23,'ge', delay=pms.delay_4TO2)
# for i in range(N_e):
#     counter = 0
#     while counter < pms.sparse_e4TOe2 * pms.N_e_23:
#         j = rn.randint(0, N_e - 1)
#         dist_effect = ((1.0 / math.sqrt(2 * math.pi * (sigma_ee_2from4 ** 2))) * \
#                        math.exp(-((distance_ee(j, i)) ** 2 / (2 * (sigma_ee_2from4 ** 2)))))
#         if rn.random() < dist_effect and C_ee42[i, j] == 0 and \
#                         abs(orientation_Ex(i) - orientation_Ex(j)) != 90:  # no autopses
#             if (orientation_Ex(i) - orientation_Ex(j) == 0):
#                 C_ee42[i, j] =  2 * wi_eTOe
#             else:
#                 C_ee42[i, j] =  1*wi_eTOe
#             counter += 1

# C_ei42 = Connection(G_e, G_i23, 'ge', delay=pms.delay_4TO2)
# for i in range(N_e) :
#     counter = 0
#     while counter < pms.sparse_e2TOi2 *pms.N_i_23 :
#         j = rn.randint(0,N_i-1)
#         dist_effect  = (1.0 / math.sqrt(2 * math.pi * (sigma_ei ** 2))) * \
#                        math.exp(-((distance_ei(i,j)) ** 2 / (2 * (sigma_ei ** 2))))
#         # facilitating projections towards inhibitory neurons of the other eye
#         if rn.random() < dist_effect and C_ei[i,j] == 0:
#             if (orientation_Ex(i)-orientation_In(j) == 0) :
#                 C_ei[i,j] = 5 * wi_eTOi
#                 counter = counter - 1
#             elif abs(orientation_Ex(i)-orientation_In(j)) == 90 :
#                 # orthogonal grtings with small weight, broader innrvation of inh. neurons that exc. neurons
#                 C_ei[i,j] = 0 * wi_eTOi
#                 #counter = counter - 1
#             else :
#                 C_ei[i,j] =  0.5* wi_eTOi
#             counter += 1

# ########################## BACK FROM LAYER 23
# C_i23toe4 = Connection(G_i23,G_e,'gi',delay = pms.delay_2TO4)
# for i in range(N_i) :
#     counter = 0
#     while counter < sparseness_inh *N_e :
#         j = rn.randint(0,N_e-1)
#         dist_effect  = ((1.0 / math.sqrt(2 * math.pi * (sigma_ie ** 2))) * \
#                        math.exp(-((distance_ie(i,j)) ** 2 / (2 * (sigma_ie ** 2))))) / (3*abs(islefteye(i)-islefteye_In(j))+1)
#         # facilitates projections towards excitator neurons of the same eye
#         if (rn.random() < dist_effect and C_ie[i,j] == 0):
#             if islefteye_In(i) - islefteye(j) == 0 : # same eye, 1-1 or 0-0 :
#                 if abs(orientation_In(i) - orientation_Ex(j)) == 0 :
#                     #C_ie[i, j] = 0 * wi_iTOe
#                     C_ie[i, j] = 5 * wi_iTOe
#                 elif abs(orientation_In(i) - orientation_Ex(j)) == 45 or abs(orientation_In(i) - orientation_Ex(j)) == 135 :
#                     C_ie[i,j] = 5 * wi_iTOe
#                 else :
#                     counter = counter - 1
#             else :
#                 if orientation_In(i) - orientation_Ex(j) == 0 :
#                     #C_ie[i, j] =  0 * wi_iTOe
#                     C_ie[i, j] = 5 * wi_iTOe
#                 elif abs(orientation_In(i) - orientation_Ex(j)) == 45 or abs(orientation_In(i) - orientation_Ex(j)) == 135 :
#                     C_ie[i,j] = 2 * wi_iTOe
#                 else :
#                     counter = counter - 1
#             #     # TODO: This aspect is curcial, so main target for changes
#             counter += 1

print('Starting connections within layer 2/3...')
######################### CONNECTIONS WITHIN LAYER 23 ##################################################################
# excitatory E->E connections layer 2
# C_ee2 = Connection(G_e23, G_e23, 'ge', delay=delay_eTOe)
# for i in range(pms.N_e_23):
#     counter = 0
#     while counter < pms.sparse_e2TOe2 * pms.N_e_23:
#         j = rn.randint(0, pms.N_e_23 - 1)
#         dist_effect = ((1.0 / math.sqrt(2 * math.pi * (sigma_ee_2to2 ** 2))) * \
#                        math.exp(-((distance_ee(i, j)) ** 2 / (2 * (sigma_ee_2to2 ** 2)))))
#         if rn.random() < dist_effect and i != j and C_ee2[i, j] == 0 and abs(orientation_Ex(i) - orientation_Ex(j)) != 90:  # no autopses
#             if (orientation_Ex(i) - orientation_Ex(j) == 0):
#                 C_ee2[i, j] =  3 * wi_eTOe
#             else:
#                 C_ee2[i, j] =  0.5*wi_eTOe
#             counter += 1
#
# print('1')
#################################  excitatory E->I connections Layer 2
# C_ei2 = Connection(G_e23,G_i23,'ge',delay= delay_eTOi)
# for i in range(N_e) :
#     counter = 0
#     while counter < pms.sparse_e2TOi2 *pms.N_i_23 :
#         j = rn.randint(0,N_i-1)
#         if distance_ei(i, j) > 5: # only long range projection
#             dist_effect = ((1.0 / math.sqrt(2 * math.pi * (sigma_ei ** 2))) * \
#                            math.exp(-((distance_ei(i, j)) ** 2 / (2 * (sigma_ei ** 2)))))
#             # facilitating projections towards inhibitory neurons of the other eye
#             if rn.random() < dist_effect and C_ei2[i, j] == 0:
#                 if (orientation_Ex(i) - orientation_In(j) == 0):
#                     # TODO: Broader exictation of inhibitory neurons anad broader spreading
#                     C_ei2[i, j] = 0 * wi_eTOi
#                     counter = counter - 1
#                 elif abs(orientation_Ex(i) - orientation_In(j)) == 90:
#                     # orthogonal grtings with small weight, broader innrvation of inh. neurons that exc. neurons
#                     C_ei2[i, j] = 5 * wi_eTOi
#                     # counter = counter - 1
#                 else:
#                     C_ei2[i, j] = 0.5 * wi_eTOi
#                 counter += 1
#
# print('2')
# # inihibtory I->I connections layer 4 - without further details
# C_ii2 = Connection(G_i23,G_i23,'gi',delay= delay_iTOi,weight=wi_iTOi,sparseness = pms.sparse_i2TOi2)
#
# print('3')
# #inhibitory I-> E connections layer 4
# C_ie2 = Connection(G_i23,G_e23,'gi',delay = delay_iTOe)
# for i in range(N_i) :
#     counter = 0
#     while counter < pms.sparse_i2TOe2 * pms.N_e_23 :
#         j = rn.randint(0,N_e-1)
#         dist_effect  = (1.0 / math.sqrt(2 * math.pi * (sigma_ie ** 2))) * \
#                        math.exp(-((distance_ie(i,j)) ** 2 / (2 * (sigma_ie ** 2))))
#         # facilitates projections towards excitator neurons of the same eye
#         if (rn.random() < dist_effect and C_ie2[i,j] == 0):
#             if islefteye_In(i) - islefteye(j) == 0 : # same eye, 1-1 or 0-0 :
#                 if abs(orientation_In(i) - orientation_Ex(j)) == 0 :
#                     #C_ie[i, j] = 0 * wi_iTOe
#                     C_ie2[i, j] = 5 * wi_iTOe
#                 elif abs(orientation_In(i) - orientation_Ex(j)) == 45 or abs(orientation_In(i) - orientation_Ex(j)) == 135 :
#                     C_ie2[i,j] = 5 * wi_iTOe
#                 else :
#                     counter = counter - 1
#             else :
#                 if orientation_In(i) - orientation_Ex(j) == 0 :
#                     #C_ie[i, j] =  0 * wi_iTOe
#                     C_ie2[i, j] = 5 * wi_iTOe
#                 elif abs(orientation_In(i) - orientation_Ex(j)) == 45 or abs(orientation_In(i) - orientation_Ex(j)) == 135 :
#                     C_ie2[i,j] = 2 * wi_iTOe
#                 else :
#                     counter = counter - 1
#             #     # TODO: This aspect is curcial, so main target for changes
#             counter += 1
#
# print('Connections within layer 2/3 done')
print('Connections done.')

####################################### FEEDBACK ######################################################################

# Feedback connections from layer 6, modeled as low-conductance NMDA and smaller AMPA due to reduced bouton size
C_efe = Synapses(G_e, G_e, 'w : 1','ge += w', delay = pms.delay_4TO5* ms +pms.delay_5TO4 * ms)
C_efe.connect()
#C_ef.connect_one_to_one(weight=0.6*wi_eTOe,delay =  pms.delay_4TO5+pms.delay_5TO4)
#C_efe.connect_full(weight=0.4*wi_eTOe)
for i in range(N_e):
    for j in range(N_e):
        if i != j and abs(orientation_Ex(i) == orientation_Ex(j))  and \
                (islefteye(i) == islefteye(j)):  # no autopses
            #C_efe.connect(i = i,j = j)
            C_efe.w[i,j] = 0.1 * wi_eTOe
            #C_efe.delay[i, j] = 0.8 * pms.delay_4TO5+pms.delay_5TO4 + \
            #                    0.2 * rn.random() * pms.delay_4TO5+pms.delay_5TO4

# low conductance NMDA receptor
C_efeNMDA = Synapses(G_e, G_e,'w  :1', 'geNMDA += w', delay = pms.delay_4TO5 * ms +pms.delay_5TO4 * ms)
C_efeNMDA.connect()
for i in range(N_e):
    for j in range(N_e):
        if C_efe.w[i,j][0] != 0 and C_efeNMDA.w[i,j][0] == 0.0:
            #C_efeNMDA.connect(i = i, j = j)
            C_efeNMDA.w[i,j] = 0.05 * wi_eTOe
            #C_efeNMDA.delay[i, j] = 0.8 * pms.delay_4TO5+pms.delay_5TO4 + \
            #                    0.2 * rn.random() * pms.delay_4TO5+pms.delay_5TO4

###################################### PLASTICITY MECHANISMS ##########################################################

if SN_ON :
    print('Normalizing excitatory to inhibitory weights...')
    for i in xrange(N_i):
        sum_in = C_ei.w[:, i].sum()
        if not sum_in == 0:
            for j in xrange(N_e) :
                if C_ei.w[j,i][0] != 0.0:
                    C_ei.w[j,i] = total_in_eTOi * C_ei.w[j, i] / sum_in
                    C_ei_NMDA.w[j,i] = 3 * C_ei.w[j,i]
                    
    print('Normalizing excitatory to excitatory weights...')
    for i in xrange(N_e):
        sum_in = C_ee.w[:, i].sum()
        if not sum_in == 0:
            for j in xrange(N_e) :
                if C_ee.w[j,i][0] != 0.0:
                    C_ee.w[j,i] = total_in_eTOe * C_ee.w[j, i] / sum_in
                    C_ee_NMDA.w[j,i] = 3 * C_ee.w[j,i]     
                    
# =============================================================================
#     print('Normalizing inhibitory to excitatory weights...')
#     for i in xrange(N_e):
#         sum_in = C_ie.w[:, i].sum()
#         if not sum_in == 0:
#             for j in xrange(N_i) :
#                 if C_ie.w[j,i][0] != 0.0:
#                     C_ie.w[j,i] = total_in_iTOe * C_ie.w[j, i] / sum_in
#                     C_ie_GABAB.w[j,i] = 0.2 * C_ie.w[j,i]                      
# =============================================================================
                     
if STP_ON :
    print('Setting STP..')
    STP_ee = STP(C_ee,taud=600*ms,tauf=21*ms,U=0.53)            # depressing synapse
    STP_efe = STP(C_efe, taud=600 * ms, tauf=21 * ms, U=0.53)   # depressing synapse
    #STP_ie = STP(C_ie,taud=1900*ms,tauf=210*ms,U=0.2) # depressing synapse
    print('STP done.')

if SRA_ON :
    def SpikeRateAdaptation(spikes):
        '''Changes g_sra for neurons which spiked at the last time step'''
        if defaultclock.t > run_time/(3) and Exercise:
            G_e.gsra[spikes] += 0.9 * pms.g_sra
        else:
            G_e.gsra[spikes] += pms.g_sra
            #G_e.gmcurr[spikes] += pms.g_mcurr
    #SRA = SpikeMonitor(G_e, record=True, function=SpikeRateAdaptation)

if STDP_ON :
    print('Setting STDP...')
    # excitatory STDP
    eqs_stdp_e = '''
    dApe/dt=-Ape/taupre : 1
    dAde/dt=-Ade/taupost : 1

    '''
    #stdp_eTOe = STDP(C_ee, eqs=eqs_stdp_e, pre='Ape += Ap; w+= Ade',
    #                 post='Ade +=Ad; w+=Ape', wmin=0, wmax=total_in_eTOe)
    stdp_eTOe2 = STDP(C_ee42, eqs=eqs_stdp_e, pre='Ape += Ap; w+= Ade',
                     post='Ade +=Ad; w+=Ape', wmin=0, wmax=total_in_eTOe)
    # stdp_eTOi = STDP(C_ei, eqs=eqs_stdp_e, pre='Ape += Ap; w+= Ade',
    #                      post='Ade +=Ad; w+=Ape', wmin=0, wmax=total_in_eTOe)


    #inhibitory STDP
    eqs_stdp_i = '''
    dApi/dt=-Api/tauprei : 1
    dApi2/dt=-Api2/tauprei : 1
    '''
    tdp_iTOe = STDP(C_ie, eqs=eqs_stdp_i, pre='Api += ampl; w+= Api2-alpha',
     post='Api2 +=ampl; w+=Api', wmin=0, wmax=total_in_iTOe)
    tdp_iTOe_GABA = STDP(C_ie_GABA, eqs=eqs_stdp_i, pre='Api += ampl; w+= Api2-alpha',
     post='Api2 +=ampl; w+=Api', wmin=0, wmax=total_in_iTOe)    
    #stdp_iTOi = STDP(C_ii, eqs=eqs_stdp_i, pre='Api += ampl; w+= Api2-alpha',
    #post='Api2 +=ampl; w+=Api', wmin=0, wmax=total_in_iTOi)

    # different E -> I
    # stdp_eTOi = STDP(C_ei, eqs=eqs_stdp_i, pre='Api += ampl_eTOi; w+= Api2',
    # post='Api2 +=ampl_eTOi; w+=Api+alpha', wmin=0, wmax=total_in_eTOi)
    print('STDP done.')


            #C_ee.W[:, i] = total_in_eTOe * C_ee.W[:, i] / sum_in
    # for i in xrange(N_e):
    #     sum_in = C_ie.W[:, i].sum()
    #     if not sum_in == 0:
    #         C_ie.W[:, i] = total_in_iTOe * C_ie.W[:, i] / sum_in
    # for i in xrange(N_i):
    #     sum_in = C_ii.W[:, i].sum()
    #     if not sum_in == 0:
    #         C_ii.W[:, i] = total_in_iTOi * C_ii.W[:, i] / sum_in
    # for i in xrange(N_i):
    #     sum_in = C_ei.W[:, i].sum()
    #     if not sum_in == 0:
    #         C_ei.W[:, i] = total_in_eTOi * C_ei.W[:, i] / sum_in
    # for i in xrange(1):
    #     sum_in = C_ee42.W[:, i].sum()
    #     if not sum_in == 0:
    #         C_ee42.W[:, i] = 5*total_in_eTOi * C_ee42.W[:, i] / sum_in

if GAPS_ON:
    @network_operation(when='end', clock=EventClock(dt = fast_steps))
    def gapjunctions() :
        for i in range(N_i) :
            for j in range(N_i) :
                if (C_iigap[i,j] != 0) :
                    G_i[j].Igap += gapweight * ((G_i[i].V - El) - (G_i[j].V - El))
                    G_i[i].Igap -= gapweight * ((G_i[i].V - El) - (G_i[j].V - El))

if Exercise :
    print("Exercise!")
    
if Kesten :
    @network_operation(when='end', dt = fast_steps)
    def kesten() :
        for i in range(N_e) :
            for j in range(N_e) :
                if (C_ee.w[i,j][0] != 0.0)  :
                    epskesten = np.random.normal(0.9923 * 0.001,0.05* 0.001) #np.random.normal(pms.kestenepsmed,pms.kestenepsstd,N_e)
                    mukesten = np.random.normal(0.0077*0.001,0.03* 0.001) #np.random.normal(pms.kestenmumed,pms.kestenmedstd,N_e)
                    C_ee.w[i,j] = epskesten * C_ee.w[i,j] + mukesten
                    C_ee_NMDA.w[i,j] = epskesten * C_ee_NMDA.w[i,j] + mukesten * vunit

# weight recording for all connections, only important ones are inhibitory to excitatory
if Record_weights :
    times_W = []
    index_before = -1
    W_eTOe = np.zeros((int(floor(run_time/middle_steps)),N_e,N_e))
    W_eTOi = np.zeros((int(floor(run_time/middle_steps)),N_e,N_i))
    W_iTOe = np.zeros((int(floor(run_time/middle_steps)),N_i,N_e))
    W_iTOi = np.zeros((int(floor(run_time/middle_steps)),N_i,N_i))
    @network_operation(when='end', dt = middle_steps)
    def record_weights():
        global index_before
        index_new = int(floor(defaultclock.t / middle_steps))
        if index_new == index_before:
            index_new += 1
        index_before = index_new;
        W_eTOe[index_new] = C_ee.w[:,:]
        W_eTOi[index_new] = C_ei.w[:,:]
        W_iTOe[index_new] = C_ie.w[:,:]
        W_iTOi[index_new] = C_ii.w[:,:]
        #W_eTOe42[index_new] = C_ee42.w
        times_w.append(defaultclock.t/middle_steps)  ## plot of the orientation distribution

############################################# PLOTS FOR CONNECTIVITY ################################################
# nx, ny = 20, 10
# x = [x for x in range(nx)]
# y = [z for z in range(ny)]
# vx, vy = meshgrid(x, y)
# z = [orientation_Ex(x) for x in range(N_e)]
# fig = figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(vx, vy, z, c=z, cmap=cm.rainbow)  # ,
#
# nx, ny = 10, 5
# x = [x for x in range(nx)]
# y = [z for z in range(ny)]
# vx, vy = meshgrid(x, y)
# z = [orientation_In(x) for x in range(N_i)]
# fig = figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(vx, vy, z, c=z, cmap=cm.rainbow)

# plot of connections relatice to a specific target
# nx, ny = 10,5
# x = [x for x in range(nx)]
# y = [z for z in range(ny)]
# vx,vy = meshgrid(x,y)
# z = C_ei.W[5,:].todense()
# z = [x*1000 for x in z]
# fig = figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(vx,vy,z )#, cmap=cm.Spectral)
# nx, ny = 20,10
# x = [x for x in range(nx)]
# y = [z for z in range(ny)]
# vx,vy = meshgrid(x,y)
# z = C_ie.W[3,:].todense()
# z = [x*1000 for x in z]
# fig = figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(vx,vy,z )#, cmap=cm.Spectral)
# figure()
# show()

#ax = fig.add_subplot(111, projection='3d')

###################### thalamocortical input ##########################################################################

fraction_thTOi = 1 # how many input traings are shared with inhibitory neurons, important to keep up activity when one
                # population is suppressed

# different types of input, correlated or uncorrelated with increase or decrease for specific periods
if input_mode == 'poisson' :
    G_tcl = PoissonGroup(N_th, rates=rand(N_th) * 10 * input_strength * Hz)
    G_tcr = PoissonGroup(N_th, rates=rand(N_th) * 10 * input_strength * Hz)
    G_tcl_toIn = G_tcl[:N_th/fraction_thTOi]
    G_tcr_toIn = G_tcr[:N_th/fraction_thTOi]
    C_thl = Connection(G_tcl, G_e, 'ge', sparseness=1.0, delay=pms.delay_the_e, weight=2*wi_eTOe)
    C_thr = Connection(G_tcr, G_e, 'ge', sparseness=1.0, delay=pms.delay_the_e, weight=2*wi_eTOe) # SAME INPUT
    C_thil = Connection(G_tcl_toIn, G_i, 'ge', sparseness = 1.0, delay = pms.delay_the_i,weight=1*wi_eTOi)
    C_thir = Connection(G_tcr_toIn, G_i, 'ge', sparseness = 1.0, delay = pms.delay_the_i,weight=1*wi_eTOi) # SAME INPUT
elif input_mode == 'corr_sep' :
    G_tcl = HomogeneousCorrelatedSpikeTrains(N_th,r=10*input_strength*Hz,c=0.2,tauc=5*ms)
    G_tcl_toIn = G_tcl[:N_th/fraction_thTOi]
    G_tcr = HomogeneousCorrelatedSpikeTrains(N_th,r=10*input_strength*Hz,c=0.2,tauc=5*ms)
    G_tcr_toIn = G_tcr[:N_th/fraction_thTOi]
    C_thl = Connection(G_tcl, G_e, 'ge', sparseness=1.0, delay=pms.delay_the_e, weight=2*wi_eTOe)
    C_thr = Connection(G_tcr, G_e, 'ge', sparseness=1.0, delay=pms.delay_the_e, weight=2*wi_eTOe) # SAME INPUT
    C_thil = Connection(G_tcl_toIn, G_i, 'ge', sparseness = 1.0, delay = pms.delay_the_i,weight=1*wi_eTOi)
    C_thir = Connection(G_tcr_toIn, G_i, 'ge', sparseness = 1.0, delay = pms.delay_the_i,weight=1*wi_eTOi) # SAME INPUT
elif input_mode == 'corr_equal' :
    G_tcl = HomogeneousCorrelatedSpikeTrains(N_th,r=10*input_strength*Hz,c=0.2,tauc=2*ms)
    G_tcl_toIn = G_tcl[:N_th/fraction_thTOi]
    C_thl = Connection(G_tcl, G_e, 'ge', sparseness=1.0, delay=pms.delay_the_e, weight=1*wi_eTOe)
    C_thr = Connection(G_tcl, G_e, 'ge', sparseness=1.0, delay=pms.delay_the_e, weight=1*wi_eTOe) # SAME INPUT
    C_thil = Connection(G_tcl_toIn, G_i, 'ge', sparseness = 1.0, delay = pms.delay_the_i,weight=1*wi_eTOi)
    C_thir = Connection(G_tcl_toIn, G_i, 'ge', sparseness = 1.0, delay = pms.delay_the_i,weight=1*wi_eTOi) # SAME INPUT
elif input_mode == 'complex' :
    times_left,times_right = Input.createSpiketrains(pms.N_th,10 * input_strength,int(run_time/(1000*ms)),0.25,2,0,Patching)
    indices = [x for x,z in enumerate(times_left) for y,y2 in enumerate(z) if y2 != 0]
    times = [y for x,z in enumerate(times_left) for y,y2 in enumerate(z) if y2 != 0] * ms
    G_tcl = SpikeGeneratorGroup(N_th,indices, times)
    indices = [x for x,z in enumerate(times_right) for y,y2 in enumerate(z) if y2 != 0]
    times = [y for x,z in enumerate(times_right) for y,y2 in enumerate(z) if y2 != 0] * ms    
    G_tcr = SpikeGeneratorGroup(N_th, indices, times)
    C_thl = Synapses(G_tcl, G_e, 'w : 1','ge += w')
    C_thl.connect(p = 0.05)
    C_thl.w = wi_eTOe
    C_thr = Synapses(G_tcr, G_e, 'w : 1','ge += w')
    C_thr.connect(p = 0.05)
    C_thr.w = wi_eTOe
    C_thil = Synapses(G_tcl, G_i, 'w : 1','ge += w')
    C_thil.connect(p = 0.05)
    C_thil.w = wi_eTOi    
    C_thir = Synapses(G_tcr, G_i, 'w : 1','ge += w')
    C_thir.connect(p = 0.05)
    C_thir.w = wi_eTOi    
elif input_mode == 'inc_contrast' :
    times_left,times_right = Input.increaseContrast(pms.N_th,10 * input_strength,int(run_time/(1000*ms)),0.25,2,1.5,2)
    #G_tcl = SpikeGeneratorGroup(N_th,[(x,y*ms) for x,z in enumerate(times_left) for y,y2 in enumerate(z) if y2 != 0],sort=True)
    #G_tcr = SpikeGeneratorGroup(N_th, [(x,y*ms) for x,z in enumerate(times_right) for y,y2 in enumerate(z) if y2 != 0], sort=True)
    #G_tcl_toIn = G_tcl[:N_th/fraction_thTOi]
    #G_tcr_toIn = G_tcr[:N_th/fraction_thTOi]
    #C_thl = Connection(G_tcl, G_e, 'ge', delay=True, sparseness = 0.5, weight = wi_eTOe)
    #C_thr = Connection(G_tcr, G_e, 'ge', delay=True, sparseness = 0.5, weight = wi_eTOe)
    #C_thil = Connection(G_tcl_toIn, G_i, 'ge', delay = True,sparseness = 0.5,weight = wi_eTOi)
    #C_thir = Connection(G_tcr_toIn, G_i, 'ge', delay = True, sparseness = 0.5, weight = wi_eTOi)
    indices = [x for x,z in enumerate(times_left) for y,y2 in enumerate(z) if y2 != 0]
    times = [y for x,z in enumerate(times_left) for y,y2 in enumerate(z) if y2 != 0] * ms
    G_tcl = SpikeGeneratorGroup(N_th,indices, times)
    indices = [x for x,z in enumerate(times_right) for y,y2 in enumerate(z) if y2 != 0]
    times = [y for x,z in enumerate(times_right) for y,y2 in enumerate(z) if y2 != 0] * ms    
    G_tcr = SpikeGeneratorGroup(N_th, indices, times)
    C_thl = Synapses(G_tcl, G_e, 'w : 1','ge += w')
    C_thl.connect(p = 0.05)
    C_thl.w = wi_eTOe
    C_thr = Synapses(G_tcr, G_e, 'w : 1','ge += w')
    C_thr.connect(p = 0.05)
    C_thr.w = wi_eTOe
    C_thil = Synapses(G_tcl, G_i, 'w : 1','ge += w')
    C_thil.connect(p = 0.05)
    C_thil.w = wi_eTOi    
    C_thir = Synapses(G_tcr, G_i, 'w : 1','ge += w')
    C_thir.connect(p = 0.05)
    C_thir.w = wi_eTOi        

# only stimulate specific neurons with the input, 0 degree left and 90 degree right eye e.g.
for i in range(N_th):
    for j in range(N_e):
        if orientation_Ex(j) == 0 and C_thl[i,j] != 0 and islefteye(j):  # orientation_Ex(j) != 0 or not islefteye(j): :
            C_thl.w[i, j] = 3 * wi_eTOe
            C_thl.delay[i,j] = 0.75 *  pms.delay_the_e* second + 0.25 * rn.random() * pms.delay_the_e * second
        #elif C_thl[i, j] != 0 :
         #   C_thl.w[i,j] = 0 #wi_eTOe
for i in range(N_th):
    for j in range(N_e):
        if orientation_Ex(j) == 90  and C_thr[i,j] != 0 and not islefteye(j):
            C_thr.w[i, j] = 3 * wi_eTOe
            C_thr.delay[i, j] = 0.75 *  pms.delay_the_e* second + 0.25 * rn.random() * pms.delay_the_e * second
        #elif C_thr[i, j] != 0 :
         #   C_thr.w[i,j] = 0 #wi_eTOe

# how_many = 0
# for x in range(N_th) :
#     for n in range(N_e) :
#         if C_thl[x,n] != 0:
#             how_many += 1
#         if C_thr[x,n] != 0 :
#             how_many += 1
#     print(how_many)
#     how_many = 0
# Noisy input towards PV+
#Th_Bkg = PoissonGroup(N_th, rates=rand(N_th*10) * 10 * input_strength * Hz)
#C_thir_general = Connection(Th_Bkg, G_i, 'ge', sparseness = 0.10, delay = pms.delay_the_i+1.0,weight=1*wi_eTOi)

# # SWITCHED TARGETS for inhibition, meaning that inh. inhibiting 90 degree right receives input with is similar to that one
# of excitatory neurons 0 degree left
for i in range(N_th/fraction_thTOi):
    for j in range(N_i):
        if orientation_In(j) == 90 and C_thil[i,j] != 0 and not islefteye_In(j):# orientation_Ex(j) != 0 or not islefteye(j): :
            C_thil.w[i, j] =  3 * wi_eTOi
            C_thil.delay[i, j] = 0.75 * pms.delay_the_i* second + 0.25 * rn.random() * pms.delay_the_i * second
        elif orientation_In(j) == 0 and C_thil[i,j] != 0 and islefteye_In(j) :
            C_thil.w[i,j] = 3 * wi_eTOi
            C_thil.delay[i, j] = 0.75 * pms.delay_the_i* second + 0.25 * rn.random() * pms.delay_the_i* second
for i in range(N_th/fraction_thTOi):
    for j in range(N_i):
        if orientation_In(j) == 0 and C_thir[i,j] != 0 and islefteye_In(j):
            C_thir.w[i, j] = 3 * wi_eTOi
            C_thir.delay[i, j] = 0.75 * pms.delay_the_i* second + 0.25 * rn.random() * pms.delay_the_i* second
        elif orientation_In(j) == 90 and C_thir[i,j] != 0 and not islefteye_In(j):
            C_thir.w[i, j] = 3 * wi_eTOi
            C_thir.delay[i, j] = 0.75 * pms.delay_the_i* second + 0.25 * rn.random() * pms.delay_the_i* second

#background input, no correlation between exc. and inh. input and large sizes with
#    low sparseness to avoid possible summating effects - currently not implemented
G_Bg6 = PoissonGroup(N_Bg, rates=5 * Hz)
C_Be = Synapses(G_Bg6, G_e, 'w : 1','ge += w') #, sparseness=0.15, delay=delay_eTOe, weight=0.5*wi_eTOe) # from layer 6
C_Be.connect( p = 0.15)
C_Be.w = 0.5*wi_eTOe
G_Bg2 = PoissonGroup(N_Bg, rates=3 * Hz)
C_Bi = Synapses(G_Bg2, G_i, 'w : 1','ge += w')
C_Bi.connect(p = 0.1)#, sparseness=0.1, delay=delay_eTOi, weight=0.5*wi_eTOi) # from layer 2/3
C_Bi.w = 0.5*wi_eTOi


################################ Monitors for recording and plotting along with some testmonitors ####################
testspikes = SpikeMonitor(G_tcl)
testspikes2 = SpikeMonitor(G_tcr)
#test2 = SpikeMonitor(G_e)
#NMDA_Curr = StateMonitor(G_e,'geNMDA', record=0)
#AMPA_Curr = StateMonitor(G_e,'ge', record=0)
#testmembrane = StateMonitor(G_e,'V',record=0)
#Gapcurrent = StateMonitor(G_i,'Igap',record=0)
#M = SpikeMonitor(G_e,record=True)
#M2 = SpikeMonitor(G_i, record=True)
#Th = SpikeMonitor(G_tcl, record=True)

#SpikeMonitorRight = PopulationRateMonitor(G_e[2:7])
#SpikeMonitorLeft = PopulationRateMonitor(G_e[194:199])
Dings = SpikeMonitor(G_e, record=True)
Dings2 = SpikeMonitor(G_i, record=True)
# M2 = SpikeMonitor(G_tc)
#Mei = StateMonitor(G_e, 'gi', record=True)
#Mee = StateMonitor(G_e, 'ge', record=True)
#Layer2_Membrane = StateMonitor(G_Layer2,'V',record=True)
#Spikes_e = SpikeMonitor(G_e)
#Spikes_i = SpikeMonitor(G_i)
#Test = StateMonitor(G_e, 'gi', record=True)
#Test2 = StateMonitor(G_e, 'gib', record=True)
#Gapcurrent = StateMonitor(G_i,'Igap',record=True)
if simple_weights :
    Weights_A_left0Degree = StateMonitor(C_ie,'w',record =  C_ie[G_i000l,G_e000l], dt = fast_steps) # C_ie[:,G_e000l]
    Weights_A_right90Degree = StateMonitor(C_ie,'w',record =  C_ie[G_i090r,G_e090r], dt = fast_steps) # C_ie[:,G_e090r]
#Weights_B_left0Degree = StateMonitor(C_ie_GABAB,'w',record = C_ie_GABAB[:,G_e000l], dt = fast_steps)
#Weights_B_right90Degree = StateMonitor(C_ie_GABAB,'w',record = C_ie_GABAB[:,G_e090r], dt = fast_steps)

# =============================================================================
# AMPAC = StateMonitor(G_e,'ge',record = True)
# NMDAC = StateMonitor(G_e,'geNMDA',record = True)
# =============================================================================

BeforeAfter_Ex = SpikeMonitor(G_e, record = True)
BeforeAfter_In = SpikeMonitor(G_i, record = True)
runcounter = 0
run(run_time, report='text')

if simple_weights :
    y = Weights_A_right90Degree[ C_ie[G_i090r,G_e090r]].w
    plot(Weights_A_right90Degree.t / ms, [mean([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0])) ])
    mean_weights =  [mean([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0]))]
    std_weights =  [np.std([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0])) ]
    plt.fill_between(Weights_A_right90Degree.t/ ms,map(add,mean_weights,std_weights),map(sub,mean_weights,std_weights),alpha=0.3)
    figure()
    y = Weights_A_left0Degree[ C_ie[G_i000l,G_e000l]].w
    mean_weights =  [mean([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0])) ]
    std_weights =  [np.std([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0])) ]    
    plot(Weights_A_left0Degree.t / ms, [mean([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0]))])
    plt.fill_between(Weights_A_right90Degree.t/ms,map(add,mean_weights,std_weights),map(sub,mean_weights,std_weights),alpha=0.3)
    figure()
    y = Weights_A_right90Degree[ C_ie[G_i090r,G_e090r]].w
    plot(Weights_A_right90Degree.t / ms, [mean([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0])) ])
    mean_weights =  [mean([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0]))]
    std_weights =  [np.std([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0])) ]
    plt.fill_between(Weights_A_right90Degree.t/ ms,map(add,mean_weights,std_weights),map(sub,mean_weights,std_weights),alpha=0.3)    
    y = Weights_A_left0Degree[ C_ie[G_i000l,G_e000l]].w
    mean_weights =  [mean([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0])) ]
    std_weights =  [np.std([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0])) ]    
    plot(Weights_A_left0Degree.t / ms, [mean([x[z] for x in y if x[0] > 0.0]) for z in range(len(y[0]))])
    plt.fill_between(Weights_A_right90Degree.t/ms,map(add,mean_weights,std_weights),map(sub,mean_weights,std_weights),alpha=0.3)    
#y = Weights_B_left0Degree[ C_ie_GABAB[:,G_e000l]].w
#plot(Weights_B_left0Degree.t / ms, [mean([x[z] for x in y]) for z in range(len(y[0])) ])
#y = Weights_B_right90Degree[ C_ie_GABAB[:,G_e090r]].w
#plot(Weights_B_right90Degree.t / ms, [mean([x[z] for x in y]) for z in range(len(y[0])) ])


############################################ PLOTS ###############################################################

#plot(Gapcurrent[0])
#plot(Test)
#Correlations between input spike trains
#plot(NMDA_Curr[0])
# corrs = list()
# for x in range(N_th) :
#     for y in range(x+1,N_th) :
#         corrs.append(total_correlation(testspikes.spiketimes[x], testspikes.spiketimes[y]))
# print(mean(corrs))
# corrs = list()
# for x in range(N_th) :
#     for y in range(x,N_th) :
#         corrs.append(total_correlation(testspikes.spiketimes[x], testspikes2.spiketimes[y]))
# print(mean(corrs))
# corrs = list()
# for x in range(N_th) :
#     for y in range(x+1,N_th) :
#         corrs.append(total_correlation(testspikes2.spiketimes[x], testspikes2.spiketimes[y]))
# print(mean(corrs))
#figure()
#plot(Test[2])
# raster_plot(testspikes)
#figure()
#plot(Test2[2])
# print(test2[1])
# print(len(test2[1]))

def plotWeights() :
    """Plot weight changes over time for connectuons towards neurons prefering the same or different orientations"""
    fig = figure()
    tempbuffer = list()
    gs = gridspec.GridSpec(2,2)
    ax00 = fig.add_subplot(gs[0, 0])
    for y in G_i090r:
        for z in G_e090r:
            if W_iTOe[0][y, z] != 0 and orientation_In(y) == orientation_Ex(z) :
                for x in range(0, len(W_iTOe)):
                    tempbuffer.append(W_iTOe[x][y, z])
                if not len(tempbuffer) == 0:
                    if orientation_In(y) == orientation_Ex(z):
                        same90r = plot([x for x in range(0, len(W_iTOe))], tempbuffer, 'k')
                    else:
                        different = plot([x for x in range(0, len(W_iTOe))], tempbuffer, 'g')
            tempbuffer = list()
        plt.title('I_R90 to E_R90 Layer 4')
    tempbuffer = list()
    ax00 = fig.add_subplot(gs[0, 1])
    for y in G_i000l:
        for z in G_e000l:
            if W_iTOe[0][y, z] != 0 and orientation_In(y) == orientation_Ex(z):
                for x in range(0, len(W_iTOe)):
                    tempbuffer.append(W_iTOe[x][y, z])
                if not len(tempbuffer) == 0:
                    if orientation_In(y) == orientation_Ex(z):
                        same90r = plot([x for x in range(0, len(W_iTOe))], tempbuffer, 'k')
                    else:
                        different = plot([x for x in range(0, len(W_iTOe))], tempbuffer, 'g')
            tempbuffer = list()
        plt.title('I_L0 to E_L0 Layer 4')
    tempbuffer = list()
    tempbuffer2 = list()
    ax00 = fig.add_subplot(gs[1, 0])
    for y in G_i090r:
        for x in range(0, len(W_iTOe)):
            tempbuffer2.append(mean([w for z,w in enumerate(W_iTOe[x][y, :]) if w != 0 and orientation_Ex(z) == orientation_In(y)]))
        tempbuffer.append(tempbuffer2)
        tempbuffer2 = list()
    std_weights = list()
    mean_weights = list()
    for v in range(0, len(W_iTOe)) :
        mean_weights.append(np.mean([max(x[v],0) for x in tempbuffer]))
        std_weights.append(np.std([x[v] for x in tempbuffer]))

    plt.plot([x for x in range(0, len(W_iTOe))], mean_weights)#, yerr = std_weights)
    plt.fill_between([x for x in range(0, len(W_iTOe))],map(add,mean_weights,std_weights),map(sub,mean_weights,std_weights),alpha=0.5)
    # if (input_mode == 'complex'and (Patching)) or input_mode == 'inc_contast'  :
    #     trans = mtransforms.blended_transform_factory(ax00.transData, ax00.transAxes)
    #     ax00.fill_between(x, 0, 0.1, where=x > 4000, facecolor='red', alpha=0.5, transform=trans)
    tempbuffer2 = list()
    tempbuffer = list()
    plt.title('I_R90 to E_R90 Layer 4 mean')
    ####
    ax00 = fig.add_subplot(gs[1, 1])
    for y in G_i000l:
        for x in range(0, len(W_iTOe)):
            tempbuffer2.append(mean([w for z,w in enumerate(W_iTOe[x][y, :]) if w != 0 and orientation_Ex(z) == orientation_In(y) ]))
        tempbuffer.append(tempbuffer2)
        tempbuffer2 = list()
    std_weights = list()
    mean_weights = list()
    for v in range(0, len(W_iTOe)) :
        mean_weights.append(np.mean([max(x[v],0) for x in tempbuffer]))
        std_weights.append(np.std([x[v] for x in tempbuffer]))

    plt.plot([x for x in range(0, len(W_iTOe))], mean_weights)
    plt.fill_between([x for x in range(0, len(W_iTOe))],map(add,mean_weights,std_weights),map(sub,mean_weights,std_weights),alpha=0.5)
    # if (input_mode == 'complex'and (Patching)) or input_mode == 'inc_contast'  :
    #     trans = mtransforms.blended_transform_factory(ax00.transData, ax00.transAxes)
    #     ax00.fill_between(x, 0, 0.1, where=x >4000, facecolor='green', alpha=0.5, transform=trans)
    plt.title('I_L0 to E_L0 Layer 4 mean')
    # figure()
    # for z in range(N_e) :
    #     for x in range(0, len(W_eTOe42)):
    #         tempbuffer.append(W_eTOe42[x][z, 0])
    #     if not len(tempbuffer) == 0:
    #         if islefteye(z):
    #             left = plot([x for x in range(0, len(W_eTOe42))], tempbuffer, 'b')
    #         else:
    #             right = plot([x for x in range(0, len(W_eTOe42))], tempbuffer, 'r')
    #     tempbuffer = list()

def computeDominanceTimes() :
    """Dominance durations rather than activity durations, therefore stepwise with time window X the corresponding
        dominating population depending on spike count
        Old implementation: based on membrane potential average, however biased by many spikes due to reset potential

        Depending on the stimulus, the intervall might be splitted in three parts:
        - before patching and after patching
        - before and during inc contrast

        Shorter switches in dominance of < 100ms sometimes occured, I tried to step over them if this happens.
        Otherwise, I look at which population is on average more active at a specific moment in time and,
        depending on whether it's the currently active population or not, I increase its dominance duration or save it
        and start a new one for the other eye. """
    figure()
    start = time.time()
    lefteye = [G_e000l[x] for x in rn.sample(range(1,24),5)]# G_e000l[5:10];# lefteye.extend(G_e045l); lefteye.extend(G_e090l); lefteye.extend(G_e135l)
    righteye =  [G_e090r[x] for x in rn.sample(range(1,24),5)]#G_e090r[5:10];# righteye.extend(G_e045r); righteye.extend(G_e090r); righteye.extend(G_e135r)
    done = time.time()
    left_active = 0
    right_active = 0
    step = 1
    s = 0.120
## Neue Version
#    s = 0.086
#    buffer_left = [[1.0]*(stop)] * len(lefteye)
#    buffer_right = [[1.0]*(stop)] * len(righteye)
#    dings_dict = Dings.spike_trains()
#    for count,i in enumerate(lefteye) :
#        tmp_buff = [0.0]*(stop)
#        for x in range(0, stop, step) :
#            ti = x
#            x = x * 0.001 * second	                                                                                                                                                                                                                                                                                          
#            tmp_buff[ti] = sum([1 / (sqrt(2 * pi) * s) * e ** (-0.5 * (float(y - x ) / s) ** 2) for y in dings_dict[i]])        
##        Ende = 0
##        for x in range(0, stop, 1) :
##            ti = x
##            x = x * 0.001 * second	           
##            # nur relevante Zeitschritte durchgehen
##            important_timesteps = list()  
##            for count2,m in enumerate(dings_dict[i][Ende:]) : 
##                if m > x+0.5*second :
##                    Ende = Ende + count2/2 
##                    break
##                else :
##                    important_timesteps.append(m)
##            tmp_buff[ti] = sum([1 / (sqrt(2 * pi) * s) * e ** (-0.5 * (float(y - x ) / s) ** 2) for y in important_timesteps])
#        buffer_left[count] = tmp_buff
#    for count,i in enumerate(righteye) :
#        tmp_buff = [0.0]*(stop)
#        for x in range(0, stop, step) :
#            ti = x
#            x = x * 0.001 * second	                                                                                                                                                                                                                                                                                          
#            tmp_buff[ti] = sum([1 / (sqrt(2 * pi) * s) * e ** (-0.5 * (float(y - x ) / s) ** 2) for y in dings_dict[i]])
#        buffer_right[count] = tmp_buff        
###    Alte Version
    buffer_left = [[1.0]*(stop)] * len(lefteye)
    buffer_right = [[1.0]*(stop)] * len(righteye)
    dings_dict = Dings.spike_trains()
    for count,i in enumerate(lefteye) :
        tmp_buff = [1.0]*(stop)
        Ende = 0
        for x in range(0, stop, step) :
            ti = x
            spikess = 0
            x = x * 0.001 * second                                                                                                                                                                                                                                                                                   
            for count2,y in enumerate(dings_dict[i][Ende:]) :
                if y > x and y < x+0.3*second : spikess += 1 #(1 / (sqrt(2 * pi) * s) * e ** (-0.5 * (float(y - x ) / s) ** 2))
                if y > x+0.3*second : 
                    Ende = Ende + 1
                    break
            tmp_buff[ti/step] = spikess
        buffer_left[count] = tmp_buff
    for count,i in enumerate(righteye) :
        tmp_buff = [1.0]*(stop) 
        Ende = 0
        for x in range(0, stop, step) :
            ti = x
            spikess = 0
            x = x * 0.001 * second
            for count2,y in enumerate(dings_dict[i][Ende:]) :
                if y > x and y < x+0.3*second : spikess += 1 #(1 / (sqrt(2 * pi) * s) * e ** (-0.5 * (float(y - x ) / s) ** 2))
                if y > x+0.3*second : 
                    Ende = Ende + 1
                    break
            tmp_buff[ti/step] = spikess
        #buffer_left.append([mean(Mee[i][x:x + step]) for x in range(0, len(Mee[i]) - step + 1, step)])
        buffer_right[count] = tmp_buff
###        Ende alter versuin
    print(start-done)
    dominance_times_left = []
    dominance_times_right = []
    dominance_times_left_contrast = []
    dominance_times_right_contrast = []
    left_active = 0
    right_active = 0
    print(buffer_left[0][:500])
    print(buffer_right[0][:500])
    global mainlist
    mainlist = list()
    if input_mode == 'complex' and (Patching) :
        active_o25pc = 0
        active_o50pc = 0

        active_o75pc = 0
        for moment in range(0,math.trunc(stop/(3*step))) :
            lefties = sum([x[moment] for x in buffer_left])
            righties = sum([x[moment] for x in buffer_right])
            if lefties == 0 and righties == 0 and left_active == 0 and right_active > 0:
                dominance_times_right.append(right_active)
                right_active = 0
                mainlist.append([0,0])
            elif lefties == 0 and righties == 0 and left_active > 0 and right_active == 0:
                dominance_times_left.append(left_active)
                left_active = 0
                mainlist.append([0, 0])
            elif lefties == 0 and righties == 0 :
                pass
            elif lefties >= righties and left_active > 0:
                mainlist.append([0,0])
                left_active += step
            elif lefties < righties and right_active > 0 :
                right_active += step
                mainlist.append([1, 0])
            elif lefties >= righties and left_active == 0 and \
                    (sum([x[min(moment+2,len(x))] for x in buffer_left]) >= sum([x[moment+2] for x in buffer_right])):
                if not right_active == 0 :
                    dominance_times_right.append(right_active)
                right_active = 0
                left_active += step
                mainlist.append([0,1])
            elif lefties >= righties and left_active == 0 and \
                     (sum([x[min(moment+2,len(x))] for x in buffer_left]) < sum([x[moment + 2] for x in buffer_right])):
                right_active += step
                mainlist.append([1, 0])
            elif lefties < righties and right_active == 0  and \
                    (sum([x[min(moment+2,len(x))] for x in buffer_left]) < sum([x[moment+2] for x in buffer_right])):
                if not left_active == 0 :
                    dominance_times_left.append(left_active)
                left_active = 0
                right_active += step
                mainlist.append([1, 0])
            elif lefties < righties and right_active == 0 and \
                 (sum([x[min(moment+2,len(x))] for x in buffer_left]) >= sum([x[moment + 2] for x in buffer_right])):
                mainlist.append([0,1])
                left_active += step
            if (lefties > 0.25 * righties and right_active > 0) or (righties > 0.25 * lefties and left_active > 0) :
                active_o25pc += 1
            if (lefties > 0.5 * righties and right_active) or (righties > 0.5 * lefties and left_active > 0) :
                active_o50pc += 1
            if (lefties > 0.75 * righties and right_active) or (righties > 0.75 * lefties and left_active > 0):
                active_o75pc += 1                
        if left_active > 0 : # last piece
            dominance_times_left.append(left_active)
        else :
            dominance_times_right.append(right_active)
        print("Mixed perceptions")
        print(active_o25pc)
        print(active_o50pc)
        print(active_o75pc)
        active_o25pc = 0
        active_o50pc = 0
        active_o75pc = 0        
        for moment in range(int(2*stop/(3*step)), math.trunc((stop/step))-1):
            lefties = sum([x[moment] for x in buffer_left])
            righties = sum([x[moment] for x in buffer_right])
            if lefties == 0 and righties == 0 and left_active == 0 and right_active > 0:
                dominance_times_left_contrast.append(right_active)
                right_active = 0
                mainlist.append([0,0])
            elif lefties == 0 and righties == 0 and left_active > 0 and right_active == 0:
                dominance_times_right_contrast.append(left_active)
                left_active = 0
                mainlist.append([0, 0])
            elif lefties == 0 and righties == 0 :
                pass
            elif lefties >= righties and left_active > 0:
                mainlist.append([0,0])
                left_active += step
            elif lefties < righties and right_active > 0 :
                right_active += step
                mainlist.append([1, 0])
            elif lefties >= righties and left_active == 0 and \
                    (sum([x[min(moment + 2, len(x))] for x in buffer_left]) >= sum(
                        [x[moment + 2] for x in buffer_right])):
                if not right_active == 0:
                    dominance_times_right_contrast.append(right_active)
                right_active = 0
                left_active += step
                mainlist.append([0, 1])
            elif lefties >= righties and left_active == 0 and \
                    (sum([x[min(moment + 2, len(x))] for x in buffer_left]) < sum(
                        [x[moment + 2] for x in buffer_right])):
                right_active += step
                mainlist.append([1, 0])
            elif lefties < righties and right_active == 0 and \
                    (sum([x[min(moment + 2, len(x))] for x in buffer_left]) < sum(
                        [x[moment + 2] for x in buffer_right])):
                if not left_active == 0:
                    dominance_times_left_contrast.append(left_active)
                left_active = 0
                right_active += step
                mainlist.append([1, 0])
            elif lefties < righties and right_active == 0 and \
                    (sum([x[min(moment + 2, len(x))] for x in buffer_left]) >= sum(
                        [x[moment + 2] for x in buffer_right])):
                mainlist.append([0, 1])
                left_active += step
            if (lefties > 0.25 * righties and right_active > 0) or (righties > 0.25 * lefties and left_active > 0) :
                active_o25pc += 1
            if (lefties > 0.5 * righties and right_active) or (righties > 0.5 * lefties and left_active > 0) :
                active_o50pc += 1
            if (lefties > 0.75 * righties and right_active) or (righties > 0.75 * lefties and left_active > 0):
                active_o75pc += 1                
        if left_active > 0 : # last piece
            dominance_times_left_contrast.append(left_active)
        else :
            dominance_times_right_contrast.append(right_active)
        print("Mixed perceptions")
        print(active_o25pc)
        print(active_o50pc)
        print(active_o75pc)            
        fig = figure()
        gs = gridspec.GridSpec(1, 2)
        ax00 = fig.add_subplot(gs[0, 0])
        #hist(dominance_times_left, np.linspace(0,math.trunc(max(max(dominance_times_left),max(dominance_times_right))),20), label = 'Left eye',stacked=True)#, alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))#, bins=[x for x in range(0,2000,step)])
        hist([dominance_times_right,dominance_times_left], np.linspace(0,math.trunc(max(max(dominance_times_left),max(dominance_times_right))),20), color = ['blue','red'], label = {'Right eye', 'Left eye'})#,alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))# bins=[x for x in range(0,2000, step)])
        xlabel('Dominance times [ms]')
        ylabel('Count [#]')
        title('Dominance durations before patching')
        print("Before patching")
        print(dominance_times_left)
        print(mean(dominance_times_left))
        print(dominance_times_right)
        print(mean(dominance_times_right))
        ax00 = fig.add_subplot(gs[0, 1])
        #hist(dominance_times_left, np.linspace(0,math.trunc(max(max(dominance_times_left),max(dominance_times_right))),20), label = 'Left eye',stacked=True)#, alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))#, bins=[x for x in range(0,2000,step)])
        hist([dominance_times_right_contrast,dominance_times_left_contrast], np.linspace(0,math.trunc(max(max(dominance_times_left),max(dominance_times_right))),20), color = ['blue','red'], label = {'Right eye', 'Left eye'})#,alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))# bins=[x for x in range(0,2000, step)])
        xlabel('Dominance times [ms]')
        ylabel('Count [#]')
        title('Dominance durations after patching')
        print("After Patching")
        print(dominance_times_left_contrast)
        print(mean(dominance_times_left_contrast))
        print(dominance_times_right_contrast)
        print(mean(dominance_times_right_contrast))
    elif input_mode == 'inc_contrast' :
        for moment in range(0,math.trunc((stop/3)/100)) :
            lefties = sum([x[moment] for x in buffer_left])
            righties = sum([x[moment] for x in buffer_right])
            if lefties == 0 and righties == 0 and left_active == 0 and right_active > 0:
                dominance_times_right.append(right_active)
                right_active = 0
                mainlist.append([0,0])
            elif lefties == 0 and righties == 0 and left_active > 0 and right_active == 0:
                dominance_times_left.append(left_active)
                left_active = 0
                mainlist.append([0, 0])
            elif lefties == 0 and righties == 0 :
                pass
            elif lefties >= righties and left_active > 0:
                mainlist.append([0,0])
                left_active += step
            elif lefties < righties and right_active > 0 :
                right_active += step
                mainlist.append([1, 0])
            elif lefties >= righties and left_active == 0 and \
                    (sum([x[min(moment+1,len(x))] for x in buffer_left]) >= sum([x[moment+1] for x in buffer_right])):
                if not right_active == 0 :
                    dominance_times_right.append(right_active)
                right_active = 0
                left_active += step
                mainlist.append([0,1])
            elif lefties >= righties and left_active == 0 and \
                     (sum([x[min(moment+1,len(x))] for x in buffer_left]) < sum([x[moment + 1] for x in buffer_right])):
                right_active += step
                mainlist.append([1, 0])
            elif lefties < righties and right_active == 0  and \
                    (sum([x[min(moment+1,len(x))] for x in buffer_left]) < sum([x[moment+1] for x in buffer_right])):
                if not left_active == 0 :
                    dominance_times_left.append(left_active)
                left_active = 0
                right_active += step
                mainlist.append([1, 0])
            elif lefties < righties and right_active == 0 and \
                 (sum([x[min(moment+1,len(x))] for x in buffer_left])  >=  sum([x[moment + 1] for x in buffer_right])):
                mainlist.append([0,1])
                left_active += step
            if (lefties > 0.25 * righties and right_active > 0) or (righties > 0.25 * lefties and left_active > 0) :
                active_o25pc += 1
            if (lefties > 0.5 * righties and right_active) or (righties > 0.5 * lefties and left_active > 0) :
                active_o50pc += 1
            if (lefties > 0.75 * righties and right_active) or (righties > 0.75 * lefties and left_active > 0):
                active_o75pc += 1                
        for moment in range(int(stop/300),int((2*stop/3)/100)) :
            lefties = sum([x[moment] for x in buffer_left])
            righties = sum([x[moment] for x in buffer_right])
            if lefties == 0 and righties == 0 and left_active == 0 and right_active > 0:
                dominance_times_left_contrast.append(right_active)
                right_active = 0
                mainlist.append([0,0])
            elif lefties == 0 and righties == 0 and left_active > 0 and right_active == 0:
                dominance_times_right_contrast.append(left_active)
                left_active = 0
                mainlist.append([0, 0])
            elif lefties == 0 and righties == 0 :
                pass
            elif lefties >= righties and left_active > 0:
                mainlist.append([0,0])
                left_active += step
            elif lefties < righties and right_active > 0 :
                right_active += step
                mainlist.append([1, 0])
            elif lefties >= righties and left_active == 0 and \
                    (sum([x[min(moment+1,len(x))] for x in buffer_left]) >= sum([x[moment+1] for x in buffer_right])):
                if not right_active == 0 :
                    dominance_times_right_contrast.append(right_active)
                right_active = 0
                left_active += step
                mainlist.append([0,1])
            elif lefties >= righties and left_active == 0 and \
                     (sum([x[min(moment+1,len(x))] for x in buffer_left]) < sum([x[moment + 1] for x in buffer_right])):
                right_active += step
                mainlist.append([1, 0])
            elif lefties < righties and right_active == 0  and \
                    (sum([x[min(moment+1,len(x))] for x in buffer_left]) < sum([x[moment+1] for x in buffer_right])):
                if not left_active == 0 :
                    dominance_times_left_contrast.append(left_active)
                left_active = 0
                right_active += step
                mainlist.append([1, 0])
            elif lefties < righties and right_active == 0 and \
                 (sum([x[min(moment+1,len(x))] for x in buffer_left])  >=  sum([x[moment + 1] for x in buffer_right])):
                mainlist.append([0,1])
                left_active += step
        dominance_after_left = list()
        dominance_after_right = list()
        for moment in range(int(2*stop/300), math.trunc((stop) / 100)-1):
            lefties = sum([x[moment] for x in buffer_left])
            righties = sum([x[moment] for x in buffer_right])
            if lefties >= righties and left_active > 0:
                mainlist.append([0, 1])
                left_active += step
            elif lefties < righties and right_active > 0:
                right_active += step
                mainlist.append([1, 0])
            elif lefties >= righties and left_active == 0 and \
                    (sum([x[min(moment + 1, len(x))] for x in buffer_left]) >= sum(
                        [x[moment + 1] for x in buffer_right])):
                if not right_active == 0:
                    dominance_after_right.append(right_active)
                right_active = 0
                left_active += step
                mainlist.append([0, 1])
            elif lefties >= righties and left_active == 0 and \
                    (sum([x[min(moment + 1, len(x))] for x in buffer_left]) < sum(
                        [x[moment + 1] for x in buffer_right])):
                right_active += step
                mainlist.append([1, 0])
            elif lefties < righties and right_active == 0 and \
                    (sum([x[min(moment + 1, len(x))] for x in buffer_left]) < sum(
                        [x[moment + 1] for x in buffer_right])):
                if not left_active == 0:
                    dominance_after_left.append(left_active)
                left_active = 0
                right_active += step
                mainlist.append([1, 0])
            elif lefties < righties and right_active == 0 and \
                    (sum([x[min(moment + 1, len(x))] for x in buffer_left]) >= sum(
                        [x[moment + 1] for x in buffer_right])):
                mainlist.append([0, 1])
                left_active += step
        if left_active > 0 : # last piece
            dominance_after_left.append(left_active)
        else :
            dominance_after_right.append(right_active)
        fig = figure()
        gs = gridspec.GridSpec(1, 3)
        ax00 = fig.add_subplot(gs[0, 0])
        #hist(dominance_times_left, np.linspace(0,math.trunc(max(max(dominance_times_left),max(dominance_times_right))),20), label = 'Left eye',stacked=True)#, alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))#, bins=[x for x in range(0,2000,step)])
        hist([dominance_times_right,dominance_times_left], np.linspace(0,math.trunc(max(max(dominance_times_left),max(dominance_times_right))),20), label = {'Right eye', 'Left eye'})#,alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))# bins=[x for x in range(0,2000, step)])
        xlabel('Dominance times [ms]')
        ylabel('Count [#]')
        title('Dominance durations outside increased contrast')
        print(dominance_times_left)
        print(mean(dominance_times_left))
        print(dominance_times_right)
        print(mean(dominance_times_right))
        ax00 = fig.add_subplot(gs[0, 1])
        #hist(dominance_times_left, np.linspace(0,math.trunc(max(max(dominance_times_left),max(dominance_times_right))),20), label = 'Left eye',stacked=True)#, alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))#, bins=[x for x in range(0,2000,step)])
        hist([dominance_times_right_contrast,dominance_times_left_contrast], np.linspace(0,math.trunc(max(max(dominance_times_left),max(dominance_times_right))),20), label = {'Right eye', 'Left eye'})#,alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))# bins=[x for x in range(0,2000, step)])
        xlabel('Dominance times [ms]')
        ylabel('Count [#]')
        title('Dominance durations during increased contrast')
        ax00 = fig.add_subplot(gs[0, 2])
        # hist(dominance_times_left, np.linspace(0,math.trunc(max(max(dominance_times_left),max(dominance_times_right))),20), label = 'Left eye',stacked=True)#, alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))#, bins=[x for x in range(0,2000,step)])
        hist([dominance_after_right, dominance_after_left],
             np.linspace(0, math.trunc(max(max(dominance_after_left), max(dominance_after_right))), 20),
             label={'Right eye',
                    'Left eye'})  # ,alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))# bins=[x for x in range(0,2000, step)])
        xlabel('Dominance times [ms]')
        ylabel('Count [#]')
        title('Dominance durations after increased contrast')
        print(dominance_times_left_contrast)
        print(mean(dominance_times_left_contrast))
        print(dominance_times_right_contrast)
        print(mean(dominance_times_right_contrast))
    else :
        # Mixrd perceptions for x percent activity between the groups
        active_o25pc = 0
        active_o50pc = 0
        active_o75pc = 0

        for moment in range(0,math.trunc((stop)/100)) :
            lefties = sum([x[moment] for x in buffer_left])
            righties = sum([x[moment] for x in buffer_right])
            if lefties == 0 and righties == 0 and left_active == 0 and right_active > 0:
                dominance_times_right.append(right_active)
                right_active = 0
                #mainlist.append([0,0])
            elif lefties == 0 and righties == 0 and left_active > 0 and right_active == 0:
                dominance_times_left.append(left_active)
                left_active = 0
                #mainlist.append([0, 0])
            elif lefties == 0 and righties == 0 :
                pass
            elif lefties > righties and left_active > 0:
                #mainlist.append([0,0])
                left_active += step
            elif lefties < righties and right_active > 0 :
                right_active += step
                #mainlist.append([1, 0])
            elif lefties >= 2 * righties and left_active == 0 and \
                    (sum([x[min(moment+1,len(x)-1)] for x in buffer_left]) >= 2* sum([x[min(moment+1,len(x)-1)] for x in buffer_right])):
                if not right_active == 0 :
                    dominance_times_right.append(right_active)
                right_active = 0
                left_active += step
                #mainlist.append([0,1])
            elif lefties >= 2 * righties and left_active == 0 and \
                     (sum([x[min(moment+1,len(x)-1)] for x in buffer_left]) < sum([x[min(moment+1,len(x)-1)] for x in buffer_right])):
                right_active += step
                #mainlist.append([1, 0])
            elif 2 * lefties < righties and right_active == 0  and \
                    (2 * sum([x[min(moment+1,len(x)-1)] for x in buffer_left]) < sum([x[min(moment+1,len(x)-1)] for x in buffer_right])):
                if not left_active == 0 :
                    dominance_times_left.append(left_active)
                left_active = 0
                right_active += step
                #mainlist.append([1, 0])
            elif 2 * lefties < righties and right_active == 0 and \
                 ( sum([x[min(moment+1,len(x)-1)] for x in buffer_left]) >= sum([x[min(moment+1,len(x)-1)] for x in buffer_right])):
                #mainlist.append([0,1])
                left_active += step
            if (lefties > 0.25 * righties and right_active > 0) or (righties > 0.25 * lefties and left_active > 0) :
                active_o25pc += 1
            if (lefties > 0.5 * righties and right_active) or (righties > 0.5 * lefties and left_active > 0) :
                active_o50pc += 1
            if (lefties > 0.75 * righties and right_active) or (righties > 0.75 * lefties and left_active > 0):
                active_o75pc += 1
        #plt.plot(mainlist)
        if left_active > 0 :
            dominance_times_left.append(left_active)
        else :
            dominance_times_right.append(right_active)
        print(dominance_times_left)
        print(mean(dominance_times_left[1:]))
        print(dominance_times_right)
        print(mean(dominance_times_right[1:]))
        #hist(dominance_times_left, np.linspace(0,math.trunc(max(max(dominance_times_left),max(dominance_times_right))),20), label = 'Left eye',stacked=True)#, alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))#, bins=[x for x in range(0,2000,step)])
        hist([dominance_times_left,dominance_times_right], np.linspace(0,math.trunc(max(max(dominance_times_left),max(dominance_times_right))),20), label = {'Right eye', 'Left eye'})#,alpha = 0.5)#, bins = np.linspace(0,len(Mee[0])/step,step))# bins=[x for x in range(0,2000, step)])
        xlabel('Dominance times [ms]')
        ylabel('Count [#]')
        print(active_o25pc)
        print(active_o50pc)
        print(active_o75pc)

computeDominanceTimes()
if (Record_weights) : plotWeights()

if True :
    """Firing rates before and after patching"""
    before = list()
    after = list()
    fig = figure()
    gs = gridspec.GridSpec(2,2)
    ax00 = fig.add_subplot(gs[0,0])
    BeforeAfter_Ex = Dings.spike_trains()
    BeforeAfter_In = Dings2.spike_trains()
    for neuron in G_e000l :
        before.append(len([t for t in BeforeAfter_Ex[neuron] if t < stop/3 * ms])/(stop/3000))
        after.append(len([t for t in BeforeAfter_Ex[neuron] if t > (2*stop) / 3 * ms]) / (stop / 3000))
    print(before)
    print(after)
    barplot = plt.bar([1.5,2.5],[np.mean(before),np.mean(after)],yerr=[np.std(before),np.std(after)],error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    title('Firing rates exc neurons 0 left')
    ylabel('Firing rate [Hz]')
    xlabel('Time period [a.u.]')
    plt.xticks([1,2],['Before','After'])
    before = list()
    after = list()
    ax00 = fig.add_subplot(gs[0,1])
    for neuron in G_e090r :
        before.append(len([t for t in BeforeAfter_Ex[neuron] if t < stop/3 * ms])/(stop/3000))
        after.append(len([t for t in BeforeAfter_Ex[neuron] if t > (2*stop) / 3 * ms] ) / (stop / 3000))
    print(before)
    print(after)
    barplot = plt.bar([1.5,2.5],[np.mean(before),np.mean(after)],yerr=[np.std(before),np.std(after)],error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    title('Firing rates exc neurons 90 right')
    ylabel('Firing rate [Hz]')
    xlabel('Time period [a.u.]')
    plt.xticks([1,2],['Before','After'])
    before = list()
    after = list()
    # INHIBITION
    ax00 = fig.add_subplot(gs[1,0])
    for neuron in G_i000l :
        before.append(len([t for t in BeforeAfter_In[neuron] if t < stop/3 * ms])/(stop/3000))
        after.append(len([t for t in BeforeAfter_In[neuron] if t > (2*stop) / 3 * ms]) / (stop / 3000))
    print(before)
    print(after)
    barplot = plt.bar([1.5,2.5],[np.mean(before),np.mean(after)],yerr=[np.std(before),np.std(after)],error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    title('Firing rates inh neurons 0 left')
    ylabel('Firing rate [Hz]')
    xlabel('Time period [a.u.]')
    plt.xticks([1,2],['Before','After'])
    before = list()
    after = list()
    ax00 = fig.add_subplot(gs[1,1])
    for neuron in G_i090r :
        before.append(len([t for t in BeforeAfter_In[neuron] if t < stop/3 * ms])/(stop/3000))
        after.append(len([t for t in BeforeAfter_In[neuron] if t > (2*stop) / 3* ms] ) / (stop / 3000))
    print(before)
    print(after)
    barplot = plt.bar([1.5,2.5],[np.mean(before),np.mean(after)],yerr=[np.std(before),np.std(after)],error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    title('Firing rates inh neurons 90 right')
    ylabel('Firing rate [Hz]')
    xlabel('Time period [a.u.]')
    plt.xticks([1,2],['Before','After'])

def rasterDataLayer4():
    """Sort spikes in a eye- and orientation-wise fashion and plot a scatter-plot of their activitiy along with
    a sorted display of firing rates in Hz."""
    fig = figure()
    gs = gridspec.GridSpec(2,4)
    ax00 = fig.add_subplot(gs[0, :])
    spikebuffer_e = list()
    spikebuffer_i = list()
    xs = list()
    ys = list()
    zs = list()
    M_dict = Dings.spike_trains()
    # excitazory neurons
    for consec, x in enumerate(G_e000l):
        xs.extend(M_dict[x]) ; ys.extend([consec] * len(M_dict[x])); zs.extend([orientation_Ex(x)]*len(M_dict[x]))
        spikebuffer_e.append(Dings.count[x])
    scatter(xs, ys, color='red', edgecolors='none')
    xs = list(); ys = list()
    print('12.5% plotting done')
    for consec, x in enumerate(G_e045l):
        consec += 25
        xs.extend(M_dict[x]) ; ys.extend([consec] * len(M_dict[x])); zs.extend([orientation_Ex(x)]*len(M_dict[x]))
        spikebuffer_e.append(Dings.count[x])
    scatter(xs, ys, color='green', edgecolors='none')
    xs = list(); ys = list()
    for consec, x in enumerate(G_e090l):
        consec += 50
        xs.extend(M_dict[x]) ; ys.extend([consec] * len(M_dict[x])); zs.extend([orientation_Ex(x)]*len(M_dict[x]))
        spikebuffer_e.append(Dings.count[x])
    scatter(xs, ys, color='blue', edgecolors='none')
    xs = list(); ys = list()
    for consec, x in enumerate(G_e135l):
        consec += 75
        xs.extend(M_dict[x]) ; ys.extend([consec] * len(M_dict[x])); zs.extend([orientation_Ex(x)]*len(M_dict[x]))
        spikebuffer_e.append(Dings.count[x])
    scatter(xs, ys, color='magenta', edgecolors='none')
    xs = list(); ys = list()
    print('50% plotting done')
    for consec, x in enumerate(G_e000r):
        consec += 100
        xs.extend(M_dict[x]) ; ys.extend([consec] * len(M_dict[x])); zs.extend([orientation_Ex(x)]*len(M_dict[x]))
        spikebuffer_e.append(Dings.count[x])
    scatter(xs, ys, color='red', edgecolors='none')
    xs = list(); ys = list()
    for consec, x in enumerate(G_e045r):
        consec += 125
        xs.extend(M_dict[x]) ; ys.extend([consec] * len(M_dict[x])); zs.extend([orientation_Ex(x)]*len(M_dict[x]))
        spikebuffer_e.append(Dings.count[x])
    scatter(xs, ys, color='green', edgecolors='none')
    xs = list(); ys = list()
    print('75.5% plotting done')
    xs = list() # CHANGED FOR JUST TWO POPULATIONS
    ys = list()
    for consec, x in enumerate(G_e090r):
        consec += 150
        xs.extend(M_dict[x]) ; ys.extend([consec] * len(M_dict[x])); zs.extend([orientation_Ex(x)]*len(M_dict[x]))
        spikebuffer_e.append(Dings.count[x])
    scatter(xs, ys, color='blue', edgecolors='none')
    xs = list();
    ys = list()
    for consec, x in enumerate(G_e135r):
        consec += 175
        xs.extend(M_dict[x]) ; ys.extend([consec] * len(M_dict[x])); zs.extend([orientation_Ex(x)]*len(M_dict[x]))
        spikebuffer_e.append(Dings.count[x])
    scatter(xs, ys, color='magenta', edgecolors='none')
    xs = list(); ys = list()
    #scatter(xs,ys,color = 'red',edgecolors='none')
    cmapp = cm.get_cmap('gist_rainbow')
    purple_patch = mpatches.Patch(color = 'r', label='0 Degree')
    #turq_patch = mpatches.Patch(color = cmapp(90), label='45 Degree')
    yell_patch = mpatches.Patch(color = 'blue', label='90 Degree')
    #red_patch = mpatches.Patch(color = cmapp(270), label='135 Degree')
    plt.legend(handles=[purple_patch,yell_patch])
    plt.title('eTOe: '+str(wi_eTOe)+' - ' + 'eTOi: '+str(wi_eTOi)+' - ' + 'iTOe: '+str(wi_iTOe)+' - ' + 'iTOi: '+\
              str(wi_iTOi)+ ' Input strength: '+str(input_strength) + ' - STDP: '+ str(STDP_ON) + \
              ' - STP: '+str(STP_ON)+' - SN:' + str(SN_ON) + ' - SRA:'+str(SRA_ON) + ' - '+input_mode)
    xlabel('Simulation time [seconds]')
    ylabel('Neuron number')
    ax00 = fig.add_subplot(gs[1, :2])
    spikebuffer_Hz_e = [x/ (run_time/(1000*ms)) for x in spikebuffer_e]
    std_Hz = [np.std(spikebuffer_Hz_e[x:x+25]) for x in range(0,200,25)]
    mean_Hz = [np.mean(spikebuffer_Hz_e[x:x+25]) for x in range(0,200,25)]
    #plt.bar([x for x in range(0,N_e)],spikebuffer_Hz_e)
    barplot = plt.bar([x for x in range(0,8)],mean_Hz,1,yerr=std_Hz, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    barplot[0].set_color('red')
    barplot[1].set_color('green')
    barplot[2].set_color('blue')
    barplot[3].set_color('magenta')
    barplot[4].set_color('red')
    barplot[5].set_color('green')
    barplot[6].set_color('blue')
    barplot[7].set_color('magenta')
    title('Firing rates excitatory populations')
    ylabel('Firing rate [Hz]')
    xlabel('Neuron number')
    mean =  np.mean(spikebuffer_Hz_e)
    #title('Firing rates, Average :' +str(spikebuffer_Hz_e)+ ' Hz')
    # inhibitory neurons
    xs = list();
    ys = list()
    zs = list()
    std_Hz = list()
    mean_Hz = list()
    M2_dict = Dings2.spike_trains()
    for consec, x in enumerate(G_i000l):
        xs.extend(M2_dict[x]) ; ys.extend([consec] * len(M2_dict[x])); zs.extend([180]*len(M2_dict[x]))
        spikebuffer_i.append(Dings2.count[x]/(run_time/(1000*ms)))
    print('12.5% plotting done')
    std_Hz.append(np.std(spikebuffer_i[:6]))
    mean_Hz.append(np.mean(spikebuffer_i[:6]))
    for consec, x in enumerate(G_i045l):
        consec += len(G_i000l)
        xs.extend(M2_dict[x]) ; ys.extend([consec] * len(M2_dict[x])); zs.extend([180]*len(M2_dict[x]))
        spikebuffer_i.append(Dings2.count[x] / (run_time / (1000 * ms)))
    std_Hz.append(np.std(spikebuffer_i[6:12]))
    mean_Hz.append(np.mean(spikebuffer_i[6:12]))
    for consec, x in enumerate(G_i090l):
        consec += len(G_i000l) + len(G_i045l)
        xs.extend(M2_dict[x]) ; ys.extend([consec] * len(M2_dict[x])); zs.extend([180]*len(M2_dict[x]))
        spikebuffer_i.append(Dings2.count[x] / (run_time / (1000 * ms)))
    std_Hz.append(np.std(spikebuffer_i[12:18]))
    mean_Hz.append(np.mean(spikebuffer_i[12:18]))
    for consec, x in enumerate(G_i135l):
        consec += len(G_i000l) + len(G_i045l) + len(G_i090l)
        xs.extend(M2_dict[x]) ; ys.extend([consec] * len(M2_dict[x])); zs.extend([180]*len(M2_dict[x]))
        spikebuffer_i.append(Dings2.count[x] / (run_time / (1000 * ms)))
    std_Hz.append(np.std(spikebuffer_i[18:25]))
    mean_Hz.append(np.mean(spikebuffer_i[18:25]))
    print('50% plotting done')
    for consec, x in enumerate(G_i000r):
        consec += len(G_i000l) + len(G_i045l) + len(G_i090l) + len(G_i135l)
        xs.extend(M2_dict[x]) ; ys.extend([consec] * len(M2_dict[x])); zs.extend([90]*len(M2_dict[x]))
        spikebuffer_i.append(Dings2.count[x] / (run_time / (1000 * ms)))
    std_Hz.append(np.std(spikebuffer_i[25:31]))
    mean_Hz.append(np.mean(spikebuffer_i[25:31]))
    for consec, x in enumerate(G_i045r):
        consec += len(G_i000l) + len(G_i045l) + len(G_i090l) + len(G_i135l) + len(G_i000r)
        xs.extend(M2_dict[x]) ; ys.extend([consec] * len(M2_dict[x])); zs.extend([90]*len(M2_dict[x]))
        spikebuffer_i.append(Dings2.count[x] / (run_time / (1000 * ms)))
    std_Hz.append(np.std(spikebuffer_i[31:37]))
    mean_Hz.append(np.mean(spikebuffer_i[31:37]))
    print('75.5% plotting done')
    for consec, x in enumerate(G_i090r):
        consec += len(G_i000l) + len(G_i045l) + len(G_i090l) + len(G_i135l) + len(G_i000r) + len(G_i045r)
        xs.extend(M2_dict[x]) ; ys.extend([consec] * len(M2_dict[x])); zs.extend([90]*len(M2_dict[x]))
        spikebuffer_i.append(Dings2.count[x] / (run_time / (1000 * ms)))
    std_Hz.append(np.std(spikebuffer_i[37:43]))
    mean_Hz.append(np.mean(spikebuffer_i[37:43]))
    for consec, x in enumerate(G_i135r):
        consec += len(G_i000l) + len(G_i045l) + len(G_i090l) + len(G_i135l) + len(G_i000r) + len(G_i045r) + len(G_i090r)
        xs.extend(M2_dict[x]) ; ys.extend([consec] * len(M2_dict[x])); zs.extend([90]*len(M2_dict[x]))
        spikebuffer_i.append(Dings2.count[x] / (run_time / (1000 * ms)))
    std_Hz.append(np.std(spikebuffer_i[43:]))
    mean_Hz.append(np.mean(spikebuffer_i[43:]))
    # currently no scatter plot of inhibitory neurons
    fig = figure()
    #z = [orientation_Ex(x) for x in ys]
    # ax00 = fig.add_subplot(gs[0, 2:])
    scatter(xs,ys,c=zs,cmap=cm.gist_rainbow)
    # cmapp = cm.get_cmap('gist_rainbow')
    #
    # purple_patch = mpatches.Patch(color = cmapp(0), label='0 Degree')
    # turq_patch = mpatches.Patch(color = cmapp(90), label='45 Degree')
    # yell_patch = mpatches.Patch(color = cmapp(170), label='90 Degree')
    # red_patch = mpatches.Patch(color = cmapp(270), label='135 Degree')
    # plt.legend(handles=[purple_patch, turq_patch,yell_patch,red_patch])
    # plt.legend(handles=[purple_patch, turq_patch])
    # xlabel('Simulation time [ms]')
    # ylabel('Neuron number')
    # title('Population activity inhibitory')
    figure()
    #ax00 = fig.add_subplot(gs[1, 2:])
    #spikebuffer_Hz_i = [x / (stop/1000) for x in spikebuffer_i]
    #plt.bar([x for x in range(0,N_i)],spikebuffer_Hz_i)
    #mean = np.mean(spikebuffer_Hz_i)

    #plt.bar([x for x in range(0,N_e)],spikebuffer_Hz_e)
    barplot = plt.bar([x for x in range(0,8)],mean_Hz,1,yerr=std_Hz, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    barplot[0].set_color('red')
    barplot[1].set_color('green')
    barplot[2].set_color('blue')
    barplot[3].set_color('magenta')
    barplot[4].set_color('red')
    barplot[5].set_color('green')
    barplot[6].set_color('blue')
    barplot[7].set_color('magenta')
    ylabel('Firing rate [Hz]')
    xlabel('Neuron number')
    title('Firing rates inhibitory population')

    #title('Firing rates, Average :' + str(mean)+ ' Hz')

def computeODI() :
    """Computing of ocular dominance indices based on connections rather than how much each neuron drives layer 23.
    Because layer 2/3 is currently not implemented, this is meaningless. """
    distribution = list()
    for target in range(pms.N_e_23) :
        left_driven = 0.0
        right_driven = 0.0
        #print(C_ee42)
        for source in range(pms.N_e) :
            if C_ee42[source,target] != 0 :
                # if a connection exists from layer 4 to 2
                if islefteye(source) :
                    left_driven += 1
                else :
                    right_driven += 1
        ODI = (left_driven-right_driven) / (left_driven + right_driven)
        distribution.append(ODI)
    figure()
    plt.hist(distribution)

rasterDataLayer4()
#rasterDataLayer2()
#plotConnectivity()

#computeODI()

# plot population behaviour
# fig = plt.figure()
# step = 200
# gs = gridspec.GridSpec(2,2)
# ax2 = fig.add_subplot(gs[0,:1])
# for i in G_e000l :
#     buffer = [mean(Mee[i][x:x+step]) for x in range(0,len(Mee[i])-step+1,step)]
#     plot(Mee.times[::step] / ms, buffer)
# ax2 = fig.add_subplot(gs[0,1:])
# for i in G_e090r :
#     buffer = [mean(Mee[i][x:x+step]) for x in range(0,len(Mee[i])-step+1,step)]
#     plot(Mee.times[::step] / ms,buffer)
# ax1 = fig.add_subplot(gs[1,:1])
# for i in G_e000l :
#     buffer = [mean(Mei[i][x:x+step]) for x in range(0,len(Mei[i])-step+1,step)]
#     plot(Mei.times[::step] / ms,buffer)
# ax2 = fig.add_subplot(gs[1,1:])
# for i in G_e090r :
#     buffer = [mean(Mei[i][x:x+step]) for x in range(0,len(Mei[i])-step+1,step)]
#     plot(Mei.times[::step] / ms, buffer)

#raster_plot(Th, title='Thalamic input', newfigure=True)
# raster_plot(M, title='Excitatory Layer IV', newfigure=True)
# raster_plot(M2, title='Inhibitory Layer IV', newfigure=True)
# #plt.figure()
# #raster_plot(M2,title='THalamic input')
show()

# TODO: Innervation of inhibitory from LGN??
# TODO: Push-Pull architecture
# TODO: Rivalry under normal strengths