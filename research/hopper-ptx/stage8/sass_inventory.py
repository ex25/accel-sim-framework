#!/usr/bin/env python3
"""Compiler-produced static SASS inventory; never label it a dynamic trace."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
cubin, out = map(Path, sys.argv[1:])
sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(cubin)],text=True)
counts = collections.Counter()
for line in sass.splitlines():
    match = re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Za-z0-9_.]+)\s',line)
    if match: counts[match.group(1)] += 1
result = {'origin':'compiler_static_sass','cubin_sha256':hashlib.sha256(cubin.read_bytes()).hexdigest(),
          'sass_sha256':hashlib.sha256(sass.encode()).hexdigest(), 'counts':dict(sorted(counts.items())),
          'instructions':sum(counts.values()), 'dynamic_counts_available':False}
if not any(k.startswith('HGMMA.') for k in counts) or not any(k.startswith('UTMALDG.') for k in counts):
    raise ValueError('frozen SASS lacks expected GMMA/TMA')
out.write_text(json.dumps(result,indent=2)+'\n')
print('STATIC_SASS_RESULT status=PASS instructions='+str(result['instructions']))
