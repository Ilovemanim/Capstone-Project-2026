#-------------------------------------------------------------------
#required imports
#-------------------------------------------------------------------
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from mne.io import read_raw_edf, concatenate_raws
import matplotlib.ticker as ticker
from brian2 import *
import pretty_midi

#-------------------------------------------------------------------


#-------------------------------------------------------------------
#Process EEG Data
#-------------------------------------------------------------------

#basic data information
ch_names = [
    'FP2-F4','F4-C4','C4-P4','P4-O2',
    'FP1-F3','F3-C3','C3-P3','P3-O1',
    'FP2-F8','F8-T4','T4-T6','T6-O2',
    'FP1-F7','F7-T3','T3-T5','T5-O1',
    'FZ-CZ','CZ-PZ'
]

sfreq = 256

ch_types = ['eeg'] * 18 

info = mne.create_info(
    ch_names=ch_names,
    sfreq=sfreq,
    ch_types=ch_types
)

#concatenate epoched data and saw as RAW file
patient_number = 4
folder = Path('/Users/jihoon/Desktop/Capstone 2026/SNNTraining/Epileptic EEG/P'+str(patient_number))
folder_file = []

for file in folder.iterdir():
    if file.is_file():
        folder_file.append(file)

data = np.loadtxt(folder_file[0], delimiter=',')
original_raw = mne.io.RawArray(data, info)

for inde in range(1,len(folder_file)-1):
    data = np.loadtxt(folder_file[inde],delimiter=',')
    data_raw = mne.io.RawArray(data,info)

    original_raw = concatenate_raws([original_raw, data_raw])

raw = original_raw

#PSD 
'''
fig = raw.compute_psd(fmax=70).plot(
    picks="data",
    exclude="bads",
    average=True,
    amplitude=False
)
fig.axes[0].set_title("PSD Analysis of IED EEG Recording from Epilepsy Patient P1")
plt.show()
'''

#normalize EEG data
data = raw.get_data()

data_norm = np.zeros_like(data)

for ch in range(data.shape[0]):

    tempo = data[ch]

    data_norm[ch] = (
        tempo - np.mean(tempo)
    ) / np.std(tempo)

#calculate AER
delta_diff = np.diff(data_norm, axis=1)
#-------------------------------------------------------------------





#-------------------------------------------------------------------
#create spike train
#-------------------------------------------------------------------

#threshold is std * 0.5
thresholds = np.std(delta_diff, axis=1)*0.5

#function to create spike train for channel "channel"
def get_segment_of(channel):
    events = []

    ch = ch_names.index(channel)
    thres = thresholds[ch]

    for t in range(delta_diff.shape[1]):
        d = delta_diff[ch, t]

        if d > thres:
            events.append((t,channel,1))
        elif d < -thres:
            events.append((t,channel,-1))
        else:
            events.append((t,channel,0))
    return events

#-------------------------------------------------------------------





#-------------------------------------------------------------------
#create brian SNN model
#-------------------------------------------------------------------

#basic setup
start_scope()

taupre = taupost = 20*ms
Apre = 0.01
Apost = -Apre*taupre/taupost*1.05

n = 10
N = n*n

N_E = N #ratio of E:I = 8:2
N_I = N//4

eqs = '''
dv/dt = (-v)/(100*ms) : 1 (unless refractory)
x : integer
y : integer
z : integer   # 0 = excitatory, -1 = inhibitory (geometry only)
'''

#excititory group

E = NeuronGroup(N_E, eqs,
                threshold='v>1',
                reset='v=0',
                refractory=1*ms, # 1ms refractory period
                method='euler')

E.x = 'i % n' #gives x and y position for each neuron
E.y = 'i // n'
E.z = 0
E.v = 0

#inhibitory group

I = NeuronGroup(N_I, eqs,
                threshold='v>1',
                reset='v=0',
                refractory=1*ms,
                method='euler')

I.x = 'i % 5 + 2' #inhibitory neurons are in the center of the brain
I.y = 'i // 5 + 3'
I.z = -1
I.v = 0

#synapse

#EE

S_EE = Synapses(E, E,
    '''
    w : 1
    dapre/dt = -apre/taupre : 1 (event-driven)
    dapost/dt = -apost/taupost : 1 (event-driven)
    ''',
    on_pre='''
    v_post += w
    apre += Apre
    w = clip(w + apost, 0, 2)
    ''',
    on_post='''
    apost += Apost
    w = clip(w + apre, 0, 2)
    ''')

S_EE.connect(condition='i != j',
             p='exp(-sqrt((x_pre-x_post)**2 + (y_pre-y_post)**2 + (z_pre-z_post)**2)/1.5)')

S_EE.w = "rand()" #initialization of weight

#II

S_II = Synapses(I, I,
    '''
    w : 1
    ''',
    on_pre='v_post -= w')

S_II.connect(condition='i != j',
             p='exp(-sqrt((x_pre-x_post)**2 + (y_pre-y_post)**2)/1.5)')

S_II.w = "rand()"

#EI

S_EI = Synapses(E, I,
    '''
    w : 1
    ''',
    on_pre='v_post += w')

S_EI.connect(p='exp(-sqrt((x_pre-x_post)**2 + (y_pre-y_post)**2)/1.5)')
S_EI.w = "rand()"

#IE

S_IE = Synapses(I, E,
    '''
    w : 1
    ''',
    on_pre='v_post -= w')

S_IE.connect(p='exp(-sqrt((x_pre-x_post)**2 + (y_pre-y_post)**2)/1.5)')
S_IE.w = "rand()"


#track activity

spikemon_E = SpikeMonitor(E)
spikemon_I = SpikeMonitor(I)

#visualize connection function
def visualise_connectivity(S, threshold=0.3):

    Ns = len(S.source)
    Nt = len(S.target)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # mask safely (Brian2-safe slicing)
    mask = np.array(S.w[:]) > threshold

    # -------------------------
    # Left: bipartite view
    # -------------------------
    ax[0].plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=2)
    ax[0].plot(np.ones(Nt), np.arange(Nt), 'ok', ms=2)

    for i, j in zip(np.array(S.i)[mask], np.array(S.j)[mask]):
        ax[0].plot([0, 1], [i, j], 'k-', lw=0.3)

    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(['Source', 'Target'])
    ax[0].set_ylabel('Neuron index')
    ax[0].set_xlim(-0.1, 1.1)
    ax[0].set_ylim(-1, max(Ns, Nt))

    # -------------------------
    # Right: adjacency matrix
    # -------------------------
    ax[1].plot(np.array(S.i)[mask], np.array(S.j)[mask], 'ok', ms=2)

    ax[1].set_xlim(-1, Ns)
    ax[1].set_ylim(-1, Nt)
    ax[1].set_xlabel('Source neuron index')
    ax[1].set_ylabel('Target neuron index')

    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------





#-------------------------------------------------------------------
#EEG Input On SNN Model
#-------------------------------------------------------------------
    
#calling on function for channel

ch_names = [
    'FP2-F4','F4-C4',
    'FP1-F3','F3-C3',
    'FP2-F8','F8-T4',
    'FP1-F7','F7-T3',
    'FZ-CZ',
]

ch_index = [67, 38, 62, 31, 68, 49, 61, 40, 35]
ch_index_inh = [18, 3, 16, 1, 19, 9, 15, 5, 2]

#function for segmenting data

def get_spike_time(channel_tm_segment):
    positive = []
    negative= []
    for data_value in channel_tm_segment:
        if data_value[2] == 1:
            positive.append(data_value[0])
        elif data_value[2] == -1:
            negative.append(data_value[0])

    return positive,negative

#loop

store_input = []
store_input_group = []

for channel, channel_index, inhi_channel_index in zip(ch_names,ch_index,ch_index_inh):
    
    event_channel = get_segment_of(channel) #spike train of positive and negative and zero
    event_channel_tm_segment = [e for e in event_channel if e[0] < 46079] #get 3 minute data

    positive_spike_time, negative_spike_time = get_spike_time(event_channel_tm_segment)

    input_groupE = SpikeGeneratorGroup(
        1,
        indices=[0] * len(positive_spike_time),
        times=np.array(positive_spike_time)/256 * second
    )

    input_groupI = SpikeGeneratorGroup(
    1,
    indices=[0] * len(negative_spike_time),
    times=np.array(negative_spike_time)/256 * second
)

    input_E = Synapses(input_groupE, E, on_pre='v_post += 1.5')
    input_E.connect(i=0,j=channel_index)

    input_I = Synapses(input_groupI, I, on_pre='v_post += 1.5')
    input_I.connect(i=0,j=inhi_channel_index)

    store_input_group.append(input_groupE)
    store_input_group.append(input_groupI)
    store_input.append(input_E)
    store_input.append(input_I)

#only if neccesssary
#M = StateMonitor(E, 'v', record=67) 
#-------------------------------------------------------------------
#print("SNN IED Model Synapse Weight Before Applying Music")
old_weight = np.mean(S_EE.w)


#visualize connection before running
#visualise_connectivity(S_EE,0.7)

#run simulation
run((180)*second)

new_weight = np.mean(S_EE.w)

'''
#plot simulation
mask = (spikemon_E.t > 130*second) & (spikemon_E.t < 180*second)
plot(spikemon_E.t[mask]/ms, spikemon_E.i[mask], '.k',ms=0.05)
xlabel('Time (ms)')
ylabel('Neuron index')

plt.show()

print("old weight = ",old_weight)
print("new weight = ",new_weight)
'''

#plot statemonitor of a neuron
'''
plot(M.t/ms, M.v[0], 'C0', label='Brian')

xlabel('Time (ms)')
ylabel('v')
legend()

plt.show()
'''

#visualize connection
#visualise_connectivity(S_EE,0.7)

print("old weight = ",old_weight)
print("new weight = ",new_weight)





