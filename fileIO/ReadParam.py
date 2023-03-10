import os


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

    # single .trxyt file and paramter <all> unnecessary conflict
    if len(params['data']) == 1 and params['all'] and '.trxyt' in params['data'][0]:
        params['all'] = False
    return params


def write_model_info(path: str, train_history: list, test_history: list, nb_histones: int, date: str) -> str:
    new_model_num = 0
    try:
        if os.path.isdir(path):
            contents = os.listdir(path)
            for content in contents:
                if 'model' in content:
                    model_num = int(content.split('_')[0].split('model')[-1])
                    new_model_num = max(new_model_num, model_num)
            modelname = f'model{new_model_num + 1}'
        os.mkdir(f'{path}/{modelname}')
    except:
        print('model directory creation err')
        raise Exception

    with open(f'{path}/{modelname}/log.txt', 'w') as info_file:
        info_file.write(f'{date}, number of trained h2bs:{str(nb_histones)}\n')
        info_file.write(f'train history, test history\n')
        for line_num in range(len(train_history)):
            info_file.write(f'{str(train_history[line_num])}\t{str(test_history[line_num])}\n')

    return modelname
