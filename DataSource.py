import pandas as pd


def load_data():
    df = pd.read_csv("booksummaries.csv", delimiter="\t",
                     names=['wiki_id', 'id', 'title', 'author', 'date', 'categories', 'description'])
    fdf = df.filter(items=['categories', 'description'])
    return fdf


def load_normalized_data():
    return pd.read_csv("cleanedData.csv", delimiter="\t")


def load_normalized_data_only_regex():
    return pd.read_csv("cleanedDataOnlyregex.csv", delimiter="\t")
