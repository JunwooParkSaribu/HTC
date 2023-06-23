def split_hybrid_from_others(data:dict | list):
    histones = {}
    if type(data) is dict:
        histones = data
    elif type(data) is list:
        for f in data:
            histones |= f
    else:
        raise Exception

    hybrids = {}
    others = {}
    for h2b in histones:
        if histones[h2b].get_predicted_label() == 1:
            hybrids[h2b] = histones[h2b].copy()
        else:
            others[h2b] = histones[h2b].copy()
    return hybrids, others
