import json, os, time
path = '/Users/scammermike/Downloads/substrate/artifacts/substrate/odyssey7d/v1/rehearsal/G09/A/candidate/heartbeat.json'
payload = {'activation': False, 'pid': os.getpid(), 'tick': 0}
while True:
    payload['tick'] += 1
    payload['pid'] = os.getpid()
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write('\n')
    time.sleep(0.05)
