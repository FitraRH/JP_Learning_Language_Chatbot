
# Zeta: A Japanese Language Learning Assistant

Zeta is a chatbot designed to help users learn Japanese by practicing Hiragana and Katakana. It can ask random questions about Japanese characters, check user answers, and provide explanations. It also allows users to chat with a trained model for general inquiries.

## Features
- **Japanese Practice**: The bot can ask random questions about Hiragana and Katakana, check the user's answer, and provide explanations.
- **AI-powered Suggestions**: If the user provides an incorrect answer, Zeta can ask OpenAI's GPT-3.5 for suggestions on how to improve.
- **Chat with Zeta**: Users can chat with the bot about general topics related to Hiragana and Katakana. Zeta's responses are based on the training data.

## Requirements
1. **Python 3.x**: Make sure you have Python 3.x installed on your machine.
2. **Required Libraries**: Install the required libraries using `pip`. You can install them via the `requirements.txt` file or manually.

    - `openai`
    - `pandas`
    - `numpy`
    - `tensorflow`
    - `nltk`

3. **API Key**: You will need an OpenAI API key to use GPT-3.5. You can get your API key from [OpenAI](https://beta.openai.com/signup/).

## Installation

1. Clone this repository:
    ```bash
    git clone
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have an OpenAI API key. Replace the placeholder `YOUR_API_KEY` in the script with your actual key.

4. Download the `intents_exercise.json` file containing the training data and place it in the same directory as the script.

## Usage

### Start the chatbot

Run the following command to start the chatbot:

The bot will ask you to choose an action:

- **Option 1**: Japanese Practice – You can practice by answering questions about Hiragana and Katakana.
- **Option 2**: Ask Zeta a question about Hiragana and Katakana – You can ask general questions, and Zeta will respond based on its training.

### Japanese Practice

- Type `Give me a question` to get a random question about Hiragana or Katakana.
- Type your answer, and Zeta will check it.
- If your answer is incorrect, you can request a suggestion from GPT-3.5 by typing `suggest me`.

### Ask Zeta

- Type any message, and Zeta will try to provide a relevant response.

### Exit

- Type `bye` to exit the chatbot.

## Code Structure

- **Training the Model**: The script uses TensorFlow and Keras to build and train a neural network that classifies user inputs into categories (tags) based on predefined patterns.
- **NLTK**: NLTK is used for text preprocessing, such as tokenization and lemmatization.
- **OpenAI GPT-3.5 Integration**: The script uses OpenAI's GPT-3.5 API to generate suggestions when a user gives an incorrect answer.

## Notes
- The `japanese_dataset` and `explanations` used in this project should contain data for Japanese Hiragana and Katakana learning. This data should be in the form of a dictionary with questions as keys and correct answers as values.
- Make sure to handle the API key securely and avoid pushing it to public repositories.

## License

This project is licensed under the MIT License.
