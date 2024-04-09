import datasets


def korotkov_preprocessor(dataset):
    processed = {}
    for part in ["train", "test"]:
        processed[part]["messages"] = dataset[part]["messages"]
        processed[part]["functions"] = dataset[part]["functions"]

    return processed

def mizinovmv_preprocessor(dataset):
    processed = {}
    return processed