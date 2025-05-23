
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/QBoGvLZa)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=18224143)

# HateGPT

> Fine tuning an LLM model to identify if the text contains hate speech or not.

### Dataset

- The dataset was obtained from the git repository for [hate speech and offensive language](https://github.com/t-davidson/hate-speech-and-offensive-language).

### How to run

- Make virtual environment
- Install requirements:
    ```
    pip install -r requirements.txt
    ```
    - Please note that the above requirements are for torch with Rocm support (AMD GPU). Please install the corresponding cuda libraries for NVIDIA GPU.

- Run the fine tuning script:
    ```
    python main.py
    ```