# Copyright 2020 Jaime Tierney, Adam Luchies, and Brett Byram

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the license at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and 
# limitations under the License.

import os

def read_model_params(fname):
    """Read model params from text file
    """
    f = open(fname, 'r')
    model_param_dict = {}
    for line in f:
        [key, value] = line.split(',')
        value = value.rstrip()
        if value.isdigit():
            value = int(value)
        elif value == 'None':
            value = None
        else:
            try:
                value = float(value)
            except:
                pass
        model_param_dict[key] = value  
    f.close()
    return model_param_dict



def save_model_params(fname, model_params_dict):
    """ Save model params to a text file
    """        
    f = open(fname, 'w')
    for key, value in model_params_dict.items():
        print(','.join([str(key), str(value)]), file=f)
    f.close()




def ensure_dir(path):
    """ Check if directory exists. If not, create it
    """
    if not os.path.exists(path):
        os.makedirs(path)




def add_suffix_to_path(path, suffix):
    """ Add suffix to model path
    """
    save_dir = path.split('/')[-2] + suffix
    path = path.split('/')
    path[-2] = save_dir
    path = os.path.join(*path)

    return path

