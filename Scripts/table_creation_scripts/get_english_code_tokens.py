import os
import json
from io import BytesIO
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import re
import tqdm

english_check = re.compile(r'[a-z]')

# Code to tokenize python code
def tokenize_code(code_lines):
    code_as_string = " ".join(code_lines)
    tokenized_code = tokenize(BytesIO(code_as_string.encode('utf-8')).readline)
    code_tokens = []
    unnecessary_tokens = ["\n", "", "utf-8"]
    try:
        for _, tokval, _, _, _ in tokenized_code:
            if tokval not in unnecessary_tokens:
                if(len(tokval) > 1 and (tokval.isalpha() or tokval.islower() or tokval.isupper() or english_check.match(tokval))):
                    code_tokens.append(tokval)
    except:
        return []
    return code_tokens

if __name__ == "__main__":
    code_lines = [
                "count = 0",
                "for train_idx, val_idx in kf.split(z):",
                "    count += 1",
                "    print(f\"FOLD {count}:\")",
                "    ",
                "    net = make_model()",
                "    net.fit(z[train_idx], y[train_idx], batch_size=BATCH_SIZE, epochs=850, ",
                "            validation_data=(z[val_idx], y[val_idx]), verbose=0) ",
                "    ",
                "    print(\"Train:\", net.evaluate(z[train_idx], y[train_idx], verbose=0, batch_size=BATCH_SIZE))",
                "    print(\"Val:\", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))",
                "    ",
                "    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)",
                "    print(\"Predicting Test...\")",
                "    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD"
            ]
    print("\n\nTokenized Code: \n\n", tokenize_code(code_lines))