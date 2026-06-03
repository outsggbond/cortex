import sys
import hashlib

if len(sys.argv) < 2:
    print("Error: No key provided")
    sys.exit(1)

key = sys.argv[1]
result = hashlib.md5(key.encode()).hexdigest()[:8]
print('{"secret_code": "%s"}' % result)
