def evaluate(parameter, server_id, q, trial_index):
    # send `parameter` to server based on `server_id`
    q.put({'idx': trial_index, 'res': (metric, 0))}


n_result = 0
q = mp.Queue()
ps = []

for i in range(len(servers_id)):
    parameters, trial_index = ax_client.get_next_trial()
    p = mp.Process(target=evaluate, args=(parameters, servers_id[i], q, trial_index))
    p.start()
    ps.append(p)
  
while n_result < n_trials:
    res = q.get(block=True)
    n_result += 1
    ax_client.complete_trial(trial_index=res['idx'], raw_data=res['res'])

    for i in range(len(servers_id)):
        if ps[i].is_alive() == False:
            parameters, trial_index = ax_client.get_next_trial()
            p = mp.Process(target=evaluate, args=(parameters, servers_id[i], q, trial_index))
            p.start()
            ps[i] = p
            break
