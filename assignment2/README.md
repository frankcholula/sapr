# Speech Recognition 🎙️
The second coursework assignment for EEEM030 Speech & Audio Processing & Recognition is designed to give you hands-on experience with machine learning methodology in a small development team. You'll practice key algorithms for feature extraction, model initialization, training, and testing. Throughout this assignment, you'll apply efficient recursive procedures to observe the effects of training using maximum likelihood. Additionally, you'll perform recognition for a simple, isolated-word recognition task, providing practical insight into speech recognition techniques.

## Setup
Please make sure you download the `EEEM030 Development Set 2024.zip` for all the testing audio files. The zip file should unzipped in the `assignment2` directory and rename as `dev_set` for the code to work. Unlike assignment 1, I will not be committing the zip file to the repository due to its size.

You can download the training set on the Notion Page [here](https://www.notion.so/frankcholula/SAPR-Assignment-2-Speech-Recognition-1413b40fbcd5804fa26ec6a93c12c481?pvs=4) under `Assignment Specification` in the top callout box.

This project uses Python version 3.9.6 and the dependencies are managed by Poetry.
I recommend setting up `pyenv` to manage your Python versions and `poetry` to manage your dependencies.
```bash
brew install pyenv
pyenv install 3.9.6
pyenv global 3.9.6
pip install poetry
```

Once you have the correct Python version, you can install the dependencies by
```bash
poetry shell
poetry install
```
 
# Running the code
To run the MFCC extraction, run
```bash
make extract
```
This creates a `feature_set` directory with the extracted MFCC features.
