"""


This makes random data for training and stores it in mlfinance/nlp/data


"""


import os
import random
from getpaths import getpath


def random_text(num_characters: int = None) -> str:
    """
    returns random text of length num_characters
    """

    chars = " abcdefghijklmnopqrstuvwxyz1234567890"
    char_list = [char for char in chars]

    text_list = []

    for i in range(num_characters):
        text_list.append(random.choice(char_list))

    return "".join(text_list)


def json_random_sentence() -> str:

    num_characters = 30
    labels = ["stock", "general"]

    sentence = (
        "{\n"
        + "    "
        + '"label": "'
        + random.choice(labels)
        + '",\n'
        + "    "
        + '"text": "'
        + random_text(num_characters)
        + '"\n'
        + "},\n"
    )

    return sentence


def csv_random_sentence() -> str:

    num_characters = 30
    labels = ["stock", "general"]

    sentence = (
        "" + random.choice(labels) + "," + "" + random_text(num_characters) + "\n"
    )

    return sentence


def make_json_data() -> None:
    cwd = getpath()

    data_directory = cwd / "data"

    # check if path exists
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    num_sentences = 10

    # beginning of file
    sentences = ["{\n"]

    for i in range(num_sentences):
        sentences.append(json_random_sentence())

    # end of file
    sentences.append("}")

    with open(data_directory / "train.json", "w") as file_handler:
        file_handler.write("".join(sentences))


def make_csv_data() -> None:
    cwd = getpath()

    data_directory = cwd / "data"

    # check if path exists
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    num_sentences = 10

    # beginning of file
    sentences = ["label,text\n"]

    for i in range(num_sentences):
        sentences.append(csv_random_sentence())

    with open(data_directory / "train.csv", "w") as file_handler:
        file_handler.write("".join(sentences))


if __name__ == "__main__":
    make_csv_data()
