import json, sys
r = json.load(sys.stdin)
print(r['choices'][0]['message'])
print('---TOKENS---')
for t in r['choices'][0]['logprobs']['content']:
    print(repr(t['token']), t['logprob'])
