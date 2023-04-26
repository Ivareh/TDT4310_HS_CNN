import json

config_path = './config/bertconfig.json'

def read_bert_config():
    with open(config_path, 'r') as f:
        config = json.load(f)
    bert_conf_dic = {'attention_probs_dropout_prob': config['attention_probs_dropout_prob'], \
           'hidden_act': config['hidden_act'], \
           'hidden_dropout_prob': config['hidden_dropout_prob'], \
           'hidden_size': config['hidden_size'], \
           'initializer_range': config['initializer_range'], \
           'intermediate_size': config['intermediate_size'], \
           'max_position_embeddings': config['max_position_embeddings'], \
           'num_attention_heads': config['num_attention_heads'], \
           'num_hidden_layers': config['num_hidden_layers'], \
           'type_vocab_size': config['type_vocab_size'], \
           'vocab_size': config['vocab_size']
    }
    return bert_conf_dic

def get_bert_config_value(param_name):
    bert_conf_dic = read_bert_config()
    if param_name in bert_conf_dic:
        return bert_conf_dic[param_name]
    else:
        raise ValueError('Parameter not found in BertConfig: ' + param_name, "Try reading the config file first")


print(get_bert_config_value)