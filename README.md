Analyzing the “Mozart Effect” in Epilepsy Using a Brian2 Spiking Neural Network Model
-


Overview
-
The “Mozart Effect” has been studied for decades, yet there is no unifying explanation for this phenomenon. A novel Brian2-based spiking neural network model was developed to simulate epileptic brain activity during IED (interictal epileptiform discharges). Mozart’s K448 was applied to the model through a cochlea-inspired encoding method. The results revealed variability in computational measures across patients and experimental conditions. However, raster plots of the model revealed similarities to real IED data. This research demonstrates the potential of Brian2-based SNNs for computational modeling of the epileptic brain activity.

Method
-
Data
18-channel bipolar IED EEG data from https://pubmed.ncbi.nlm.nih.gov/39920162/
Mozart K448 MIDI file from https://musescore.com/user/14896651/scores/3160916?srsltid=AfmBOorJcrgBjXAlWK1vzhRgAZ83lnzQ8AyMM3ny-rDKlR-eZZ7WpTen

Model
Brian2-based SNN with 100 excitatory neurons + 25 inhibitory neurons, with LIF and STDP mechanism

Pipeline
IED EEG data --> Spike Train Conversion (AER) --> SNN simulation (Brian2) --> Music Input Spike Train (Mozart K448) or randomized control input --> Analysis (raster plot, synaptic weight, van Rossum distance)

Results
-
SNN exhibited characteristics of IED activity, such as synchronization and hyperactivity
Mozart-derived input altered firing patterns and synaptic weights
However, Mozart K448 did not consistently outperform randomized control inputs
Results support previous findings suggesting that frequency information alone is insufficient to explain the "Mozart Effect"

Key Findings
-
Brian2-based SNNs can model aspects of epileptic EEG dynamics
Model captured some characteristics of IED activity
Frequency-based encoding of Mozart K448 was insufficient to reproduce the "Mozart Effect" computationally
Future work should incorporate other aspects of musicality, such as rhythm and harmony, and utilize more established brain models
