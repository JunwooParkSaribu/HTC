def read(file):
    file = f'{file}/params.txt'
    params = {}
    with open(file, 'r') as f:
        input = f.readlines()
        for line in input:
            line = line.strip().split('=')
            params[line[0].strip()] = int(line[1].strip())
    return params