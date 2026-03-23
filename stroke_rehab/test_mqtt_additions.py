import sys
with open('ui/neurorehab_games.html', encoding='utf-8') as f: src = f.read()
checks = [
    ('mqtt.min.js CDN',      'mqtt.min.js' in src),
    ('MQ object',            'const MQ = {' in src),
    ('connectMQTT()',        'function connectMQTT()' in src),
    ('onFusionStateChange()', 'function onFusionStateChange' in src),
    ('flashBorder()',        'function flashBorder' in src),
    ('connectMQTT() at boot','connectMQTT();' in src),
    ('hit() mqBonus',        'mqBonus' in src),
    ('MQ.connected in updSB','MQ.connected' in src),
    ('MQTT button in HTML',  'btnMQTT' in src),
]
failed = [n for n,ok in checks if not ok]
if failed: print('FAIL:', failed); sys.exit(1)
print('PASS - all additions confirmed present')
