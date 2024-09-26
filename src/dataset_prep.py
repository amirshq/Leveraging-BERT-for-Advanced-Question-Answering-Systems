# Tokenizer 
class EncodedText:
    def __init__(self,ids, offsets):
        self.ids = ids
        self.offsets = offsets
        
#Defien Tokenizer based on Model
def create_tokenizer_and_tokens(config):
    if "roberta" in config.selected_model:
        raise NotImplementedError
    elif "albert" in config.selected_model:
        raise NotImplementedError
    else:
        tokenizer = BertWordPieceTokenizer(
            MODEL_PATH[config.selected_model] + 'vocab.txt',
            lowercase = config.lowercase,
        )
        tokens = {
            'cls':tokenizer.token_to_id('[CLS]'),
            'sep':tokenizer.token_to_id('[SEP]'),
            'pad':tokenizer.token_to_id('[PAD]'),
        }
    return tokenizer,tokens

