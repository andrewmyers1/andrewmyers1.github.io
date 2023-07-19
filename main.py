import ast
import openai
import pandas as pd
import tiktoken
import gradio as gr
from openai.cli import display
from scipy import spatial

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

openai.api_key = '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'

df = pd.read_csv("trainingData.csv")

df['embedding'] = df['embedding'].apply(ast.literal_eval)


def insert_word_into_file(filepath, word_to_insert='END', words_per_insertion=400):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    words = content.split()
    modified_content = []

    for i, word in enumerate(words):
        modified_content.append(word)
        if (i + 1) % words_per_insertion == 0:
            modified_content.append(word_to_insert)

    modified_text = ' '.join(modified_content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(modified_text)


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def split_file_on_word(filepath, end_word="END"):
    # Initialize an empty list to hold our strings
    strings = []
    # Initialize an empty string to hold the current string
    current_string = ""

    # Open the file
    with open(filepath, 'r', encoding='latin-1') as f:
        # Iterate over the lines in the file
        for line in f:
            # If we find the end word, end the current string and add it to the list
            if end_word in line:
                halves = line.split(end_word)
                current_string += halves[0]
                # Add the current string to the list of strings
                strings.append(current_string.strip())
                # Start a new string
                current_string = halves[1].strip()
            else:
                # If we don't find the end word, add the line to the current string
                current_string += line

        # Add the remaining string to the list if it's not empty
        if current_string.strip():
            strings.append(current_string.strip())

    return strings


def calc_embeddings(articles):
    EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
    BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

    embeddings = []
    for batch_start in range(0, len(articles), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = articles[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end - 1}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)

    df = pd.DataFrame({"text": articles, "embedding": embeddings})
    SAVE_PATH = "trainingData.csv"
    df.to_csv(SAVE_PATH, mode='a', header=False, index=False)


def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 5
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below information on the Uluro Software to answer the subsequent question. ' \
                   'If the answer cannot be found in the information, write "I\'m not sure. Please check our Help Center ' \
                   'at: https://transtrial.zendesk.com/hc/en-us"'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nInformation on Uluro:\n"""\n{string}\n"""'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask_question(
        query: str,
        # dataframe: pd.DataFrame = df,
        # topic: str = 'Y',
        # model: str = GPT_MODEL,
        # token_budget: int = 4096 - 500,
        # print_message: bool = False,
) -> str:
    dataframe = df
    topic: str = 'Y'
    model: str = GPT_MODEL
    token_budget: int = 4096 - 500
    print_message: bool = False
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    if topic == 'Y':
        message = query_message(query, dataframe, model=model, token_budget=token_budget)
        if print_message:
            print(message)
        messages = [
            {"role": "system", "content": "You answer questions about Uluro Software with medium amount of information. "
                                          "Stop before 200 tokens. If the answer cannot be found in the articles, "
                                          "\write I\'m not sure. Please check our Help Center at: https://support."
                                          "transfrm.com/hc/en-us"},
            {"role": "user", "content": message},
        ]
    else:
        messages = [
            {"role": "system",
             "content": "You respond to questions in a polite manner"},
            {"role": "user", "content": query},
        ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=400,
        # stream=True
    )

    response_message = response["choices"][0]["message"]["content"]
    return response_message

    # i = 0
    #
    # for partial in response:
    #     if i % 20 == 0:
    #         print('')
    #
    #     if "content" in partial["choices"][0]["delta"]:
    #         print(partial["choices"][0]["delta"]["content"], end='')
    #     i += 1



if __name__ == '__main__':
    # file_path = input("\nEnter <filepath> if you would like new Data to be embedded. Else, enter 'N'\n")
    #
    # if file_path.upper() != 'N':
    #     chunks = split_file_on_word(file_path)
    #     calc_embeddings(chunks)

    # subject = input("Enter 'Y' if your question is about Uluro: ")
    # question = input("\nPlease enter your question here: ")
    #
    # ask_question(question, df, subject)
    # print("\n")

    strings, relatednesses = strings_ranked_by_relatedness("how to create a payment processor", df, top_n=10)
    for string, relatedness in zip(strings, relatednesses):
        print(f"{relatedness=:.3f}")
        display(string)

    # iface = gr.Interface(fn=ask_question,
    #                      inputs=gr.components.Textbox(lines=7, label="Enter your text"),
    #                      outputs="text",
    #                      title="Custom-trained AI Chatbot")
    #
    # iface.launch(share=True)

