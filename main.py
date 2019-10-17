import os

from experiments.proto_net import ex
from sacred.observers import MongoObserver

added_source_files = ['models/', 'models/backbones/', 'dataLoader/']

for folder in added_source_files:
    for file in os.listdir(folder):
        if file.split('.')[-1] == 'py':
            ex.add_source_file(filename=os.path.join(folder, file))

with open('mongodb_setup.txt', 'r') as f:
    try:
        x = f.read().splitlines() 
    except IOError:
        print("Could not read mongodb_setup file")
        sys.exit()

login = [i.split('=')[-1] for i in x]

ex.observers.append(MongoObserver.create(
    url = 'mongodb+srv://{}:{}@cluster0-a1qml.mongodb.net/test?retryWrites=true&w=majority'.format(login[0], login[1]),
    db_name = 'test'
))

r = ex.run()


