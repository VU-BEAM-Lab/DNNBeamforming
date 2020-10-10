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

class Logger:

    def __init__(self):
        self.entries = {}

    def __repr__(self):
        return self.__str__()

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return str(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    def append(self, path):

        epoch = len(self.entries)
        line = [epoch]
        line.append( self.entries[epoch]['loss_train'] )
        line.append( self.entries[epoch]['loss_train_eval'] )
        line.append( self.entries[epoch]['loss_val'] )
        line = [str(item) for item in line]
        line = ','.join(line)
        line += '\n'

        f = open(path, 'a')
        f.write(line)
        f.close()
