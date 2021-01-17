import importlib

def model_factory(args):
    Transformer = importlib.import_module('models.'+args.model).Transformer
    TransformerEncoder = importlib.import_module('models.'+args.model).TransformerEncoder
    TransformerDecoderLayer = importlib.import_module('models.'+args.model).TransformerDecoderLayer
    ScaledDotProductAttention = importlib.import_module('models.'+args.model).ScaledDotProductAttention
    return Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
