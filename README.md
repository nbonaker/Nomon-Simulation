Nomon Background
================
To learn more about Nomon, visit our main website [nomon.app](https://nomon.app)

Nomon, invented by [Tamara Broderick](http://people.csail.mit.edu/tbroderick/index.html), is a keyboard application that uses a single switch selection method, allowing users to select a letter or a word with a single click. These clicks are distinguished by their timing, which can be controlled depending on the users desired speed. Each letter and suggested word is paired with a set of small clocks, each one associated with each option of a letter or a word. Repeatedly, when the clock's moving hand is at noon, the user clicks the single click until the desired option is selected. Using this method, users can select words and letters to form sentences with just a single switch.

Relevant papers about Nomon include:
- Broderick, T and MacKay, DJC. Fast and flexible selection with a single switch. *PLoS ONE* 4(10), e7481. [link](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0007481)
- Broderick, T. Nomon: Efficient communication with a single switch. *Technical Report* (extension of Master's Thesis). Cavendish Laboratory, University of Cambridge. [ps](http://www.inference.org.uk/nomon/files/nomon_tech_report.ps) [pdf](http://www.inference.org.uk/nomon/files/nomon_tech_report.pdf)
- Nicholas Ryan Bonaker, Emli-Mari Nel, Keith Vertanen, and Tamara Broderick. 2022. A Performance Evaluation of Nomon: A Flexible Interface for Noisy Single-Switch Users. *CHI '22*. [link](https://dl.acm.org/doi/10.1145/3491102.3517738)
- Nicholas Bonaker, Emli-Mari Nel, Keith Vertanen, and Tamara Broderick. 2023. A Usability Study of Nomon: A Flexible Interface for Single-Switch Users. *ASSETS '23*. [link](https://doi.org/10.1145/3597638.3608415)


Nomon Simulated User Project Overview
=================
This repository contains a framework for simulating the use of nomon with data collected from real single-switch users. The repository is divided into five packages:
- **User-Simulation** (Main Package) -- Contains a framework that simulates user interactions with a running instance of the Nomon keyboard.


- **Nomon-Core** (Helper Package) -- Contains the core backend attributes that facilitate the Nomon selection mechanism
- **Nomon-Symbol** (Helper Package) -- Contains the frontend attributes needed to simulate a picture/symbol selection version of Nomon.
- **Nomon-Text** (Helper Package) -- Contains the frontend attributes needed to simualte a full-text keyboard version of Nomon. Also contains the kenlm language models used for word and character predictions in the full-text keyboard.
- **Nomon-User-Data** (Helper Package) -- Contains detailed data tables on how switch users interacted with the Nomon keyboard as they learned to use it. This data is used as an input to the Nomon-Simulation package above.

![Alt text](readme_flowchart.png?raw=true "Package Flowchart")

>***Note -- This repository contains code for a python-based implementation of the Nomon selection mechanism. We are no longer actively developing or supporting a python based application for Nomon, but a legacy application can be found [here](https://github.com/tbroderick/Nomon). We recommended checking out our [web based application](https://github.com/nbonaker/NomonWeb) (JS/HTML/PHP) if you wish to see or adapt the code for purposes beyond user simulation.*** 

Main Package
===============
User_Simulation 
------------------
### Main Files:
- **`Simulated_user_symbol.py`** -- Contains the `SimulatedUser` class that interacts with the main `Keyboard` class in the `Nomon_Symbol` module to simulate user interactions. Controls which clocks are targeted, when switch-press events occur, and saves entry data from the simulation.
- **`Simulated_user_text.py`** -- Contains the `SimulatedUser` class that interacts with the main `Keyboard` class in the `Nomon_Text` module to simulate user interactions. Controls which clocks are targeted, when switch-press events occur, and saves entry data from the simulation.
- **`SimConfig.py`** -- Hyperparameters controlling the behavior of the `SimulatedUser`
- **`run_parallel.py`** -- Allows multiple simulation instances to run concurrently on multiple cores (if available). Similar to running multiple jobs on a distributed system.
- **`simulations/*/run_sim.py`** -- Contains the `SimulationUtil` class that manages the process of running the simulation. These classes are unique to each simulation and control the varied parameters that are fed into the `SimulatedUser` and `keyboard` instances. They also manage the saving of data into the simulation's subdirectory. The `SimulationUtil` class also contains code to split parameter searches (multiple, longitudinal simulations) into jobs that can be executed concurrently on multiple CPU cores. 

### Example Simulations Directory Structure:

*examples/ \
&emsp;&emsp;&emsp;&emsp;&emsp; simulation1/ \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; results/ -- resulting csv data tables per phrase\
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; click_results/ (optional) -- resulting csv data tables per click \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; run_sim.py -- contains the SimulationUtil class \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; data_load.ipynb -- jupyter notebook to load/plot results \
&emsp;&emsp;&emsp;&emsp;&emsp; simulation2/ \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ...* 

### Running an Example Simulation:
_Simulations must be run as a python module._ For example, execute the following command from the root directory to run the "language_model_compression" simulation file: \
**`python -m User_Simulation.examples.language_model_compression.run_sim`**

### Working with Simulation Results:

#### Where Results are Saved
Results from a simulation are saved in two ways: (1) on a per-phrase basis, and (2) optionally on a per click basis.

##### Per-Phrase results
>The per-phrase results are the primary method of saving simulation results. These results contain common text-entry metrics like entry rate, click load, error rate, etc. 
Results are saved in a directory named with the datetime from when the simulation started:

*/results/sim-MM-DD-YYYY-hh-mm.csv*

##### Per-Click results (optional)
>The per-click results are an optional method of saving simulation results. These results contain a much more granular view of the nomon selection process and the posterior distribution over the selected options after each simulated click.
Results are saved in a directory named with the datetime from when the simulation started:

*/click_results/sim-MM-DD-YYYY-hh-mm.csv*

#### How Data Tables are Structured

##### Per-Phrase results
Results are saved as a long-form data table with each row containing the results from a single simulated phrase. Each phrase has the following column attributes:

>Text and Symbol Simulation Columns:
 - **user_id** -- The user_id for the click data loaded to use in the simulation.
 - **Trial Num** -- The trial number for the data. The number of trials for which a simulation is re-run is specified in the simulation's **`SimConfig`** class.
 - **Session Num** -- The session number for the user click-data used. Session Number will be set to a constant value of 1 if not replaying the user's click data in context.
 - **Phrase Num** -- Counts the number of phrases in the current session.
 - **Target Phrase** -- Can be a string or a sequence of target symbols. The target phrase for the simulated user to type.
 - **Typed Text** -- Can be a string or a sequence of symbol selections. The text typed by the simulated user. Note this may differ from the phrase text if the simulated user made an error.
 - **Click Load (clicks/selection)** -- The average number of clicks needed to make a selection for the current phrase. Calculated as _(total number of clicks) / (total number of selections made)_.
 - **Entry Rate (cpm)** -- The average number of output-selections per minute for the current phrase. An output-selection is defined as any selection that outputs text (symbol, character, punctuation, word completion). Calculated as _(total number of output-selections) / (length of time spent typing the target phrase in minutes)_.
 - **Correction Rate (%)** -- The percent of selections made in the current phrase that were a corrective action (Backspace/Undo/Clear).
 - **Error Rate (%)** -- The error rate between the typed text and target phrase. Calculated as _(The minimum edit distance between the target phrase and typed text) / (The length of the target phrase) * 100_
>Text Simulation Only Columns:
  - **Phrase Type** -- The phrase set from which the target phrase is drawn: in-vocabulary (iv) or out-of-vocabulary (oov) 
  - **Click Load (clicks/character)** -- The average number of clicks needed to type a character for the current phrase. Calculated as _(total number of clicks) / (total number of character typed)_.
  - **Entry Rate (wpm)** -- The average number of words typed per minute for the current phrase. A word is defined as 5 characters.
  - **Word Prediction Usage (%)** -- The percentage of selections made that were a word prediction.
>Simulation-Specific Columns:
>>Additional columns are added to the data table to situate the hyper-parameters used to generate the data. These colums are appended to the data table in the simulation-specific **`SimConfig`** class.
The README file in each example simulation documents the simulation-specific columns. These columns often define the independent variable a simulation is testing. Below are a few examples of what a simulation-specific column could be:
  - **Language Model Used** (language_model_compression example)
  - **Phrase Set Used** (language_model_compression example)
  - **Total Number of Word Predictions Allowed**
  - **Number of Word Predictions Allowed per Character**
  - **Number of Clocks on the Screen** (num_clocks_symbol example)
  

##### Per-Click results (optional)
Results are saved as a long-form data table with each row containing the results from a single simulated click

  - **phrase_num** -- Counts the number of phrases in the simulation
  - **selection_num** -- Counts the number of selections in the current phrase
  - **click_num** -- Counts the number of clicks in the current selection. click_num = 0 represents the prior before any clicks are made
  - **target_clock_ind** -- Index of the target clock in Nomon's backend
  - **target_clock_text** -- Text attribute of the target clock (character a-z, punctuation, backspace, undo, clear)
  - **is_post_undo** -- Boolean: is the current selection immediately following the selection of the UNDO clock
  - **num_clocks_on** -- The total number of clocks on the screen
  - **cscore_entropy** -- The entropy in bits of the posterior distribution over the clocks 
  
>Main Keys Posterior Probability Columns:
  - **a** -- model probability of the letter 'a'
  - **b** -- model probability of the letter 'b'\
  ...
  - **z** -- model probability of the letter 'z'
  - **.** (period)-- model probability of a period
  - **\'** (apostrophe) -- model probability of an apostrophe
  - **BACKSPACE** -- model probability of the backspace option
  - **CLEAR** -- model probability of the clear option
  - **UNDO** -- model probability of the undo option

>Word Options Posterior Probability Columns:
  - **word_0_cscore** --  model probability of the first available word prediction (if any)
  - **word_0_text** -- text of the first available word prediction (if any)\
  ...
  - **word_n_cscore** --  model probability of the nth available word prediction (if any)
  - **word_n_text** -- text of the nth available word prediction (if any)
  
#### How to Load Data Tables
Each example simulation has a Jupyter Notebook file `data_load.ipynb` with scripts to load and plot the data from the specific example.
By default, this script searches for all simulation results under the example's `results/` directory and concatenates them into a single, longform, Pandas DataFrame. 


Helper Packages
================

Nomon_Core
----------------
- **`BroderClocks.py`** -- Manages the `ClockInferenceEngine` and `ClockUtil` classes and communicates with the keyboard application. Handles switch-press events and makes selection decisions based on the clock probabilities. 
- **`ClockInferenceEngine.py`** -- Handles the inference and probability estimates pertaining to the clocks. 
- **`ClockUtil.py`** -- Manages the movement and position of clock hands.
- **`Config.py`** -- Hyperparameters controlling the core selection mechanism behind Nomon.

Nomon_Symbol
----------------
- **`Keyboard.py`** -- Controls the interface of the Nomon symbol keyboard. Manges the layout of the clocks/symbol pictures and keeps track of the typed text.
- **`KConfig.py`** -- Hyperparameters controlling the keyboard interface and it's layout.
- **`/resources`** -- contains the emoji characters used to initialize the symbol keyboard layout.

Nomon_Text
----------------
- **`Keyboard.py`** -- Controls the interface of the Nomon symbol keyboard. Manges the layout of the clocks/characters/word completions and keeps track of the typed text.
- **`KConfig.py`** -- Hyperparameters controlling the keyboard interface and it's layout.
- **`phrase_manager.py`** -- Contains the `Phrases` class used to load and mix the iv and oov phrase sets.
- **`/kenlm`**
    - **`kenlm_lm.py`** -- Interfaces with the kenlm models to format word and character probabilities for use in Nomon.
    - **`predictor.py`** -- Contains the `WordPredictor` class that handles queries for the word language model.
    - **`char_predictor.py`** -- Contains the `CharacterPredictor` class that handles queries for the character language model.
- **`/resources`** -- contains the kenlm files for the language models and phrase datasets for the text-entry simulation targets.

Nomon-User-Data
----------------
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
