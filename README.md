# Generative AI Assignment

## Description
Example of using Azure OpenAI to create a simple chatbot that can search documents

## Installation

### Prerequisites
- Python 3.x installed on your machine. You can download it from python.org.
- WSL
- Miniconda3

### Installing Required Packages
This project requires the installation of certain Python packages. You can install them using `pip`.

1. Clone the repository to the current folder:
    ```bash
    git clone https://github.com/NhatTranMinh15/Generative-AI-Assignment.git .
    ```
2. Create a virtual environment (optional but recommended):
    ```bash
    conda create langchain python=3.11
    ```
  Set the "langchain" env that has been just created as the running env in VS code

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Create .env file
   ```bash
  AZURE_OPENAI_ENDPOINT=https://bonbonv2-dev-eastus2.openai.azure.com
  AZURE_OPENAI_API_KEY=
  ```
