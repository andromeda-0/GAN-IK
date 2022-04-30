import os
import json

if __name__ == '__main__':
    for file_name in os.listdir('configs'):
        file_path = os.path.join('configs', file_name)
        with open(file_path) as f:
            config: dict = json.load(f)
        result_file = file_name[:-5] + '.csv'
        with open(os.path.join('results', result_file)) as f:
            data = f.readline().split(',')
            for k, v in config.items():
                data.append(str(k) + ':' + str(v))
        with open(os.path.join('results', result_file), 'w') as f:
            f.write(','.join(data))
