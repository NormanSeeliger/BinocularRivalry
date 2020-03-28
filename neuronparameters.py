# FFIM

vunit = 0.001
# 'vunit' serves at transforming all values to Volt that were previously in mV
# I did this to maintain consistency in the equations after rewriting the LIF equation, which is still in Volt,
# while maintaining clarity compared to previous versions.
# Conductance is unitless. Weight is also unitless, but volt-equivalent (not mV!!).
# thresholds, reversal potentials are in mV.

### Base parameters
alpha_val = 0.008 # 0.002 vorher
sparse_eTOe = 0.05       # recurrent excitatory sparseness
sparse_iTOe = 0.15       # inhibitory to excitatory sparseness
sparse_eTOi = 0.10       # excitatory to inhibitory  sparseness
sparse_iTOi = 0.15        # inhibitory to inhibitory sparseness
sparse_e4TOe2 = 0.09     # exc. layer 4 to exc. layer 2
sparse_e4TOi2 = 0.05     # exc. layer 4 to inh. layer 2
sparse_e4TOe5 = 0.03     # exc. layer 4 to exc. layer 2
sparse_e2TOe2 = 0.12
sparse_e2TOi2 = 0.10     # exc 23 to inhibitory 23
sparse_i2TOe2 = 0.16
sparse_i2TOi2 = 0.14
#sparse_e4TOi5 = 0.3     # exc. layer 4 to inh. layer 2
N_e = 200 #N_e=400
N_e_23 = 200            # layer 23 excitatory population
N_i_23 = 50             # layer 2/3 inhibitory population
N_i = 50 #N_i=int(0.2*N_e)
N_th = 200
N_Bg = 1
wi_eTOe = 7.4 * vunit    # max initial e->e weight (V) (AMPA+NMDA current) # von 9.2 # 5.5 mittendrin
wi_eTOi = 3 * vunit    # max initial e->i weight (V)
wi_iTOe = 10 * vunit   # max initial i->e weight (V)
wi_iTOi = 10 * vunit   # max initial i->e weight (V)
# 8.5 5 60 10
# 8.5 5 30 10 with .0025 gsra for DD of 2 secs
delay_eTOe = 1.5        # e->e latency for 1 mm (ms)
delay_eTOi = 1.0        # e->i latency for 1 mm (ms)
delay_iTOe = 0.5        # i->e latency for 1 mm (ms)
delay_iTOi = 1.0        # i->i latency for 1 mm (ms)
delay_the_i = 0.001       # thalamic disynaptic IPSP in layer 4
delay_the_e = 0.0015
# MOUSE DATA
delay_4TO2 = 0.024
delay_5TO2 = 0.037
delay_2TO4 = 0.021
delay_2TO5 = 0.028
delay_4TO5 = 2.0  # CARIANTE 30 ms
delay_5TO4 = delay_4TO5
delay_GABAB = 15

# 20 50 60 30 - 35 40 100 40 - 15 25 80 30 - 20 80 60 30 somewhat
# w/o thalamic increase: 25 100 40 30
# Neuron parameters
noise_level = 1.0 #2.25      # noise amplitude (mV) # CHANGED from 2.0
taum = 20               # membrane time constant (ms)
Vr_e = -70              # excitatory reset value  (mV)
Vr_i = -60              # inhibitory reset value  (mV)
El = -60                # resting value  (mV)
Ee = 0                  # reversal potential Excitation (mV)
Ei = -80                # reversal potential Inhibition GABA_A (mV)
Eib = -95               # reversal potential Inhibition GABA_B (mV)
taue = 3                # ms EPSP time constant (ms) AMPFA-Current
tau_nmda = 80           # NMDA current time contant NMDA-Current
taui = 10               # ms IPSP time constant (ms) GABA-Current
taub1 = 122             # time constant GABA_B (first)
taub2 = 587             # time constant GABA_B (second)
tau_sra = 996           # ms SRA time constant (ms)
g_sra =   0.004         # increment in potassium current after spike (mV)^# was 0.0025
g_mcurr = 0.0         # strength M-Current
tau_mcurr = 100         # time constant M-Current
gapweight = 0.2
# 0.0024 for 2 secs
E_k = -80               # reversal potential SRA potassium channel (M-Current)
E_dep = -60.0             # mV CHANGEDADDED
target_for_IP_e = 9     # target firing rate excitatory
target_for_IP_i = 30    # target firing rate inhibitory

# initial thesholds & voltage
Vti = 55                # minus maximum initial threshold voltage  (mV)
Vtvar = 5               # maximum initial threshold voltage swing  (mV)
Vvi = 60               # minus maximum initial voltage  (mV)
Vvar = 10               # maximum initial voltage swing  (mV)

# STDP parameters
taupre = 15             # pre-before-post STDP time constant (ms)
taupost = taupre * 2.0  # post-before-pre STDP time constant (ms)
Ap = 0.5*vunit#0.1*vunit #0.9 * vunit #4.0 * vunit        # potentiating STDP learning rate (V)
Ad = -Ap * 0.5       # depressing STDP learning rate  (V)

# iSTDP parameters
tauprei = 15            # iSTDP window time constant (ms)

balance = 1
### Synaptic normalization parameters/maximum weight
total_in_eTOe = balance*sparse_eTOe * N_e * wi_eTOe * 0.2 #1600 * vunit # (V)
total_in_iTOe = balance*sparse_iTOe * N_i * wi_iTOe * 0.25 #10000 * vunit # (V)   #-30 * mV #-12 * mV
total_in_eTOi = balance*sparse_eTOi * N_e * wi_eTOi #23200 * vunit # (V)
total_in_iTOi = balance*sparse_iTOi * N_i * wi_iTOi #12000 * vunit # (V)

# Neuronal Thresholds
V_Thresh_e = -57.3   # excitatory neurons
V_Thresh_i = -58 # (mV)    # inhibitory neurons