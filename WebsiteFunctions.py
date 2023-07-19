import datetime
import ast
import pandas as pd
import main


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_embeddings(file_path):
    chunks = main.split_file_on_word(file_path)
    main.calc_embeddings(chunks)


def get_request(prompt):
    EMBEDDINGS_PATH = "trainingData.csv"
    df = pd.read_csv(EMBEDDINGS_PATH)

    df['embedding'] = df['embedding'].apply(ast.literal_eval)

    return main.ask_question(prompt, df, 'Y')


def normal_gpt(regular_prompt):
    EMBEDDINGS_PATH = "trainingData.csv"
    df = pd.read_csv(EMBEDDINGS_PATH)

    return main.ask_question(regular_prompt, df, 'N')
