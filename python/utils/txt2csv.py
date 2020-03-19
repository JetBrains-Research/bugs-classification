import pandas as pd


def transform_tokens_to_csv(path_to_tokens, path_to_csv):
    corpus = []
    max_length = 0
    # Parse token sequences of code changes with different length
    with open(path_to_tokens) as file:
        for line in file.readlines():
            row = line.split(",")
            solution_id, tokens, cluster = row[0], row[1:-1], row[-1]
            if cluster[-1] == '\n':
                cluster = cluster[:-1]
            max_length = max(max_length, len(tokens))
            parsed = [solution_id, tokens, cluster]
            corpus.append(parsed)
    # Add paddings to align all sequences
    data = []
    for row in corpus:
        id, tokens, cluster = row[0], row[1], row[2]
        real_length = len(tokens)
        for _ in range(len(tokens), max_length):
            tokens.append('<PAD>')
        flattened_row = [id] + [real_length] + tokens + [cluster]
        data.append(flattened_row)
    # Create pandas dataframe ans save it as csv
    columns = ['id'] + ['real_len'] + [str(token_id) for token_id in range(max_length)] + ['cluster']
    df = pd.DataFrame(data=data, columns=columns)
    df.set_index(columns[0], inplace=True)
    df.to_csv(path_to_csv)
    print('Saved to ' + str(path_to_csv))
