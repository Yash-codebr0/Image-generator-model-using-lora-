import re

PAD = 0
UNK = 1

class SimpleTokenizer:
    def __init__(self, vocab_size=512, max_length=16):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.stoi = {"<pad>": PAD, "<unk>": UNK}

    def build(self, texts):
        words = []
        for text in texts:
            words.extend(re.findall(r"[a-z0-9']+", text.lower()))
        for word in sorted(set(words)):
            if len(self.stoi) >= self.vocab_size:
                break
            self.stoi[word] = len(self.stoi)

    def encode(self, text):
        words = re.findall(r"[a-z0-9']+", text.lower())
        ids = [self.stoi.get(w, UNK) for w in words[:self.max_length]]
        ids += [PAD] * (self.max_length - len(ids))
        return ids

    def state_dict(self):
        return {"stoi": self.stoi, "vocab_size": self.vocab_size, "max_length": self.max_length}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state["vocab_size"], state["max_length"])
        obj.stoi = state["stoi"]
        return obj
