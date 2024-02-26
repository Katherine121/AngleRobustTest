# sequence_test
# Background
In this project, we propose a novel angle robustness navigation paradigm to deal with flight deviation and 
design an angle robustness point-to-point navigation model for adaptively predicting direction angle.
In order to evaluate vision-based navigation methods in real scenarios, we collect a new dataset UAV_AR368 and 
design the Simulation Flight Testing Instrument (SFTI) using Google Earth, 
which can simulate real-world flight deviation effectively and avoid the cost of real flight testing.

This is the project for testing angle robustness point-to-point navigation model.

# Project Structure             
│  classify_test.py  
│  cor.py  
│  load_dataset.py  
│  load_model.py  
│  match_test.py
│  our_test.py  
│  README.md  
│  requirements.txt
│  
├─bigmap  
│  
├─checkpoint  
│  bs_models.py  
│  fsra.py  
│  lpn.py  
│  model.py  
│  rknet.py  
│  
└─utils  
    │  compress.py  
    │  compute_error.py  
    │  data_augment.py  
    │  draw.py  
    │  get_candidates.py  
    │  
    └─candidates   

# Install
`pip install -r requirements.txt`

# Prepare Datasets
Our big map dataset will be publicly available at xxxx
In this directory, we can find different big maps equipped with specified coordinates.
You should put the "bigmap" directory under the project directory.

To get all candidate images, you should run the commands below:
`cd utils
mkdir candidates
python get_candidates.py`

# Prepare Checkpoints
After training all checkpoints, you should run the commands below:
`mkdir checkpoint
mv your_model_definition_file_path checkpoint/
mv your_checkpoint_file_path checkpoint/`

# Test
For testing our angle robustness point-to-point navigation model, you should run the commands below:
`python our_test.py`
For testing classification-based navigation model, you should run the commands below:
`python classify_test.py`
For testing matching-based navigation model, you should run the commands below:
`python match_test.py`

# Citation
