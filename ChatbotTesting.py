import io
import openai # type: ignore
import random
import string # to process standard python strings
import warnings
import pandas as pd # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from japanese_dataset import japanese_dataset, explanations
from openai import OpenAI # type: ignore
client = OpenAI(api_key='')
warnings.filterwarnings('ignore')

import nltk # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
nltk.download('popular', quiet=True) # for downloading packages

data = pd.read_json("./intents_exercise.json")

bot_name = "Zeta"

words = []
classes = []
data_X = []
data_Y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        data_X.append(pattern)
        data_Y.append(intent["tag"]) ,

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

lemmatizer = WordNetLemmatizer()

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

words = sorted(set(words))
classes = sorted(set(classes))

training = []
out_empty = [0] * len(classes)

for idx, doc in enumerate(data_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
        output_row = list(out_empty)
        output_row[classes.index(data_Y[idx])] = 1
        training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_X = np.array(list(training[:, 0]))
train_Y = np.array(list(training[:, 1]))

# Split data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]), activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_Y, validation_data=(test_X, test_Y), epochs=10, verbose=1)

# Rest of your code...


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
def get_random_question():
    """Select a random question from the Japanese dataset"""
    return random.choice(list(japanese_dataset.keys()))

def get_explanation(question):
    """Get the explanation for the given question"""
    return explanations.get(japanese_dataset.get(question, ""), "")

def check_answer(question, user_answer):
    """Check if the user's answer matches the correct answer"""
    correct_answer = japanese_dataset.get(question, "")
    return user_answer.strip().lower() == correct_answer.lower()

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.5
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

0
def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        result = "Sorry, I didn't understand that. Can you please provide more information?"
    elif len(intents_list) > 1:
        result = "I'm not sure which response to provide. Can you please clarify?"
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    return result

openai.api_key = ""
def send_to_chatGPT(user_suggestion):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_suggestion}],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    content = response.choices[0].message.content.strip()
    print(content)


def main():
    while True:
        print("Zeta: My name is Vestia Zeta. I will answer your queries about Hiraganas and Katakanas. If you want to exit, type Bye!")
        print("Please choose an action:")
        print("1. Japanese practice")
        print("2. Ask me a question regarding hiragana and katakana")
        choice = input("Enter your choice: ")

        if choice == "1":
            flag = True
            while flag:
                print("Zeta: Welcome! I will help you learn some Japanese. Type 'Give me a question' to start.")
                user_input = input().strip().lower()

                if user_input == 'give me a question':
                    question = get_random_question()
                    print(f"ROBO: What's the meaning of '{question}'?")
                    user_answer = input().strip()

                    if check_answer(question, user_answer):
                        print("Zeta: Correct! Well done!")
                        print("Zeta: Type 'how is it implemented' if you want to know more about the answer.")
                    else:
                        print("Zeta: Sorry, that's not correct. The answer is:", japanese_dataset[question])
                        print("Zeta: Type suggest me if you would like me to provide you with some suggestion?")
                        user_suggestion_choice = input().lower()
                        if user_suggestion_choice == 'suggest me':
                            user_suggestion = "You're a Japanese teacher and this is your student response: " + user_answer + " to the question of " + question + ", please create a suggestion on where is the mistake on how to improve"
                            send_to_chatGPT(user_suggestion)

                elif user_input == 'how is it implemented':
                    if question:
                        explanation = get_explanation(question)
                        if explanation:
                            print("Zeta:", explanation)
                        else:
                            print("Zeta: Sorry, I don't have an explanation for that question yet.")
                    else:
                        print("Zeta: You haven't answered any question yet.")

                elif user_input == 'bye':
                    flag = False
                    print("Zeta: Dadah! Have a great day!")

                else:
                    print("Zeta: I'm sorry, I didn't understand that. Type 'Give me a question' to start.")

        elif choice == "2":
            while True:
                message = input("")
                if message == "0":
                    break
                intents = pred_class(message, words, classes)
                result = get_response(intents, data)
                print(result)

        elif choice.lower() == 'bye':
            print("Zeta: Goodbye! Have a great day!")
            break

        else:
            print("Zeta: Invalid choice. Please enter '1' or '2'. Or type 'Bye' to exit.")


if __name__ == "__main__":
    main()



