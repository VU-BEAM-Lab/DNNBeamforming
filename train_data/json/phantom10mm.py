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

#!/usr/bin/env python

import json
from pprint import pprint

if __name__ == '__main__':

    # json save file
    json_file = 'phantom10mm.json'

    # create empty data specification dictionary
    data = {}

    # train data
    data['train'] = []

    N = 9135

    target_num_list = [1,2,3,4,5,6,7,8]

    for target_num in target_num_list:
        if target_num==8:
            N=9142
        elif target_num==5:
            N=9136
        else:
            N=9135

        data['train'].append({
            'file' : f'train_data/example_phantom_10mm_70mm/chandat{target_num}.h5',
            'N' : str(N)
        })


    # train eval data
    data['train_eval'] = []

    N = 9135

    target_num_list = [1,2]

    for target_num in target_num_list:

        data['train_eval'].append({
            'file' : f'train_data/example_phantom_10mm_70mm/chandat{target_num}.h5',
            'N' : str(N)
        })

    # val data
    data['val'] = []

    N = 9131

    target_num_list = [9,10]

    for target_num in target_num_list:

        data['val'].append({
            'file' : f'train_data/example_phantom_10mm_70mm/chandat{target_num}.h5',
            'N' : str(N)
        })

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

