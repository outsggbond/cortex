from newpkg.util import answer
import sys
if answer() != 42:
    sys.exit(1)
print('ok')
