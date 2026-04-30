from . import sd3_clip


class CogVideoXT5Tokenizer(sd3_clip.T5XXLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, min_length=226)
