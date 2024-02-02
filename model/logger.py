import os.path as osp
from tensorboardX import SummaryWriter
from collections import defaultdict, OrderedDict
import numpy as np
import json

class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)

        

class Logger(object):
    def __init__(self, args, log_dir, **kwargs):
        self.args = args
        self.logger_path = osp.join(log_dir, str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+self.args.loss_type+'_scalars.json')
        self.tb_logger = SummaryWriter(
                            logdir=osp.join(log_dir, 'tflogger'),
                            **kwargs,
                        )
        self.log_config(vars(args))

        self.scalars = defaultdict(OrderedDict)
        




    def log_config(self, variant_data):
       
        config_filepath = osp.join(osp.dirname(self.logger_path), str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+self.args.loss_type+'_configs.json')
        with open(config_filepath, "w") as fd:
            json.dump(variant_data, fd, indent=2, sort_keys=True, cls=ConfigEncoder)
        

    def add_scalar(self, key, value, counter):
        assert self.scalars[key].get(counter, None) is None, 'counter should be distinct'
        self.scalars[key][counter] = value
        self.tb_logger.add_scalar(key, value, counter)


    def dump(self):
        print("Dumping File")
        with open(self.logger_path, 'w') as fd:
            json.dump(self.scalars, fd, indent=2)


    
