Nomon Simulated User Overview
================
This repository contains a framework for simulating the use of nomon with data collected from real single-switch users. The repository is divided into three sections:
- **Nomon-Core** --- Contains the core backend attributes that facilitate the Nomon selection mechanism
- **Nomon-Simulation** --- Contains a framework that simulates user interactions with a running instance of the Nomon keyboard.
- **Nomon-User-Data** --- Contains detailed data tables on how switch users interacted with the Nomon keyboard as they learned to use it. This data is used as an input to the Nomon-Simulation framework above.

***Note -- This repository contains code for a python-based implementation of the Nomon selection mechanism. We are no longer actively developing or supporting a python based application for Nomon, but a legacy application can be found [here](https://github.com/tbroderick/Nomon). We recommended checking out our [web based application](https://github.com/nbonaker/NomonWeb) (JS/HTML/PHP) if you wish to see or adapt the code for purposes beyond user simulation.*** 

Nomon Background
================
To learn more about Nomon, visit our main website [nomon.app](https://nomon.app)
Nomon, invented by [Tamara Broderick](http://people.csail.mit.edu/tbroderick/index.html), is a keyboard application that uses a single switch selection method, allowing users to select a letter or a word with a single click. These clicks are distinguished by their timing, which can be controlled depending on the users desired speed. Each letter and suggested word is paired with a set of small clocks, each one associated with each option of a letter or a word. Repeatedly, when the clock's moving hand is at noon, the user clicks the single click until the desired option is selected. Using this method, users can select words and letters to form sentences with just a single switch.

Relevant papers about Nomon include:
- Broderick, T and MacKay, DJC. Fast and flexible selection with a single switch. *PLoS ONE* 4(10), e7481. [link](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0007481)
- Broderick, T. Nomon: Efficient communication with a single switch. *Technical Report* (extension of Master's Thesis). Cavendish Laboratory, University of Cambridge. [ps](http://www.inference.org.uk/nomon/files/nomon_tech_report.ps) [pdf](http://www.inference.org.uk/nomon/files/nomon_tech_report.pdf)
- Nicholas Ryan Bonaker, Emli-Mari Nel, Keith Vertanen, and Tamara Broderick. 2022. A Performance Evaluation of Nomon: A Flexible Interface for Noisy Single-Switch Users. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems (CHI '22). Association for Computing Machinery, New York, NY, USA, Article 495, 1–17. [link](https://dl.acm.org/doi/10.1145/3491102.3517738)

Nomon-Core
================
- **`BroderClocks.py`** -- Manages the `ClockInferenceEngine` and `ClockUtil` classes and communicates with the keyboard application. Handles switch-press events and makes selection decisions based on the clock probabilities. 
- **`ClockInferenceEngine.py`** -- Handles the inference and probability estimates pertaining to the clocks. 
- **`ClockUtil.py`** -- Manages the movement and position of clock hands.
- **`Config.py`** -- Hyperparameters controlling the core selection mechanism behind Nomon.

Nomon-Simulation
================
Main Files:
- **`Keyboard.py`** -- Controls the interface of the Nomon keyboard. Manges the layout of the clocks/words/characters and keeps track of the typed text.
- **`SimulatedUser.py`** -- Interacts with the main `Keyboard` class to simulate user interactions. Controls which clocks are targeted, when switch-press events occur, and saves entry data from the simulation.
- **`SimConfig.py`** -- Hyperparameters controlling the behavior of the `SimulatedUser`
- **`KConfig.py`** -- Hyperparameters controlling the keyboard interface and it's layout.

Simulations Directory: 

*simulations/ \
&emsp;&emsp;&emsp;&emsp;&emsp; simulation1/ \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; results/ -- contains the output csv data files \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; sim.py -- contains the SimulationUtil class \
&emsp;&emsp;&emsp;&emsp;&emsp; simulation2/ \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ...* 

- **SimulationUtil** -- Simulations are managed by a SimulationUtil class. These classes are unique to each simulation and control the varied parameters that are fed into the SimulatedUser and keyboard instances. They also manage the saving of data into the simulation's subdirectory. The SimulationUtil class also contains code to split parameter searches (multiple, longitudinal simulations) into jobs that can be executed concurrently on multiple CPU cores. 

Nomon-User-Data
================
CSV data tables containing the raw interaction data for the switch-users that trialed Nomon. Each row represents a single click sent into the Nomon Keyboard. The columns in the dataset are described below:
 - **Session Num** -- The session number for the data. Sessions lasted 10 minutes each, though earlier sessions may be shorter.
 - **Phrase Num** -- Counts the number of phrases in the current session.
 - **Selection Num** -- Counts the number of selections needed to type the current phrase.
 - **Click Num** -- Counts the number of switch presses needed to make the current selection.
 - **Phrase Text** -- The target phrase presented to the user.
 - **Typed Text** -- The text currently typed by the user on a given phrase. Note this may differ from the phrase text if the user made an error.
 - **Target** -- The target word/character highlighted for the user to select.
 - **Selection** -- The word/character/corrective option ultimately selected by the user. 
 - **Clock Period (s)** -- The time in seconds it takes the clocks to make a full rotation.
 - **Click Time Relative (s)** -- The time in seconds that the user clicked their switch relative to when the clock they ultimately selected was at Noon. This value can range from [-Clock Period/2, Clock Period/2].
 - **Click Time Absolute (s)** -- The timestamp measured in seconds since epoch (epoch time) that the user clicked their switch.
 - **Dead Time (s)** -- The time in seconds since the last time the user clicked. Equal to the difference between the current and previous Click Time Absolute values. 
