def read(file):
    file = f'{file}/config.txt'
    params = {}
    with open(file, 'r') as f:
        input = f.readlines()
        for line in input:
            if '=' not in line:
                continue
            line = line.strip().split('=')
            param_name = line[0].strip()
            param_val = line[1].split('#')[0].strip()
            if param_name in ['amp', 'nChannel', 'batch_size', 'group_size', 'cut_off']:
                params[param_name] = int(param_val)
            elif param_name in ['data']:
                if param_name in params:
                    params[param_name].append(param_val)
                else:
                    params[param_name] = [param_val]
            elif param_name in ['all']:
                if param_val.lower() == 'true' or param_val.lower() == 'yes' or param_val == '1':
                    params[param_name] = True
                else:
                    params[param_name] = False
            else:
                params[param_name] = param_val
    return params
