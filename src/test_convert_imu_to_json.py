import json

def create_json_files(input_file):
    data = {}
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            time = parts[0]
            key = parts[1]
            values = list(map(float, parts[2:]))
            if key not in data:
                data[key] = []
            data[key].append({'time': time, 'w': str(values[0]), 'x': str(values[1]), 'y': str(values[2]), 'z': str(values[3])})

    for key, value in data.items():
        output_file = f'data_processing/imu_data/prep_data_q{key[-1]}.json'
        with open(output_file, 'w') as f:
            json.dump(value, f, indent=4)

if __name__ == "__main__":
    input_file = 'test_imu_data.txt'
    create_json_files(input_file)
