import json


def parse_concatenated_json(json_str: str):
    # https://stackoverflow.com/questions/36967236/parse-multiple-json-objects-that-are-in-one-line
    decoder = json.JSONDecoder()
    pos = 0
    objs = []
    while pos < len(json_str):
        json_str = json_str[pos:].strip()
        if not json_str:
            break  # Blank line case
        obj, pos = decoder.raw_decode(json_str)
        objs.append(obj)

    return objs
